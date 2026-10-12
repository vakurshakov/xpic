#include "simulation.h"

#include "src/utils/configuration.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"

namespace drift_kinetic {

static constexpr PetscReal atol = 1e-10;
static constexpr PetscReal rtol = 1e-10;
static constexpr PetscReal stol = 0;
static constexpr PetscReal divtol = PETSC_DETERMINE;
static constexpr PetscInt maxit = 1000;
static constexpr PetscInt maxf = PETSC_UNLIMITED;

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetType(snes, SNESNGMRES));
  // PetscCall(SNESSetType(snes, SNESANDERSON));
  PetscCall(SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf));
  PetscCall(SNESSetDivergenceTolerance(snes, divtol));
  PetscCall(SNESSetFunction(snes, NULL, Simulation::form_iteration, this));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMCreateGlobalVector(da, &M));
  PetscCall(DMCreateGlobalVector(da, &E_hk));
  PetscCall(DMCreateGlobalVector(da, &B_hk));
  PetscCall(DMCreateGlobalVector(da, &Bn1));
  PetscCall(DMCreateLocalVector(da, &Bn_loc));
  PetscCall(DMCreateLocalVector(da, &Bn1_loc));

  Rotor rotor(da);
  PetscCall(rotor.create_negative(&rotM));
  PetscCall(MatScale(rotB, -1));  /// @see `Simulation::form_function()`

  Region region{
    .dim = 4,
    .dof = 6,
    .start = Vector4I(0, 0, 0, 0),
    .size = Vector4I(geom_nx, geom_ny, geom_nz, 6),
  };
  PetscCall(World::create_local_dm(da, region, PETSC_COMM_WORLD, &da_EB));
  PetscCall(DMCreateGlobalVector(da_EB, &sol));

  PetscCall(init_particles(*this, particles_));

  const Configuration::json_t& json = CONFIG().json;

  bool is_gc_coord = false;
  bool is_fail_terminal = false;

  if (auto it = json.find("DriftKineticPush"); it != json.end()) {
    is_gc_coord = it->value("coord_is_gc", false);
    is_fail_terminal = it->value("fail_on_terminal_nonconvergence", false);
  }

  for (auto& sort : particles_) {
    sort->set_coord_is_gc(is_gc_coord);
    sort->set_fail_on_terminal_nonconvergence(is_fail_terminal);
  }

  LOG("  Additional settings for drift_kinetic:");
  LOG("    Use gyro-centers to set coordinates (coord_is_gc): {}", is_gc_coord);
  LOG("    Terminate on particles non-convergence (fail_on_terminal_nonconvergence): {}", is_fail_terminal);

  diagnostics_.emplace_back(std::make_unique<EnergyConservation>(*this));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(interfaces::Simulation::finalize());

  PetscCall(VecDestroy(&M));
  PetscCall(VecDestroy(&E_hk));
  PetscCall(VecDestroy(&B_hk));
  PetscCall(VecDestroy(&Bn1));
  PetscCall(VecDestroy(&Bn_loc));
  PetscCall(VecDestroy(&Bn1_loc));

  PetscCall(MatDestroy(&rotM));

  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&sol));
  PetscCall(DMDestroy(&da_EB));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::timestep_implementation(PetscInt t)
{
  PetscFunctionBeginUser;
  for (auto& sort : particles_)
    PetscCall(sort->prepare_storage());

  if (!evolve_fields) {
    PetscCall(VecCopy(E, E_hk));
    PetscCall(VecCopy(B, B_hk));
    PetscCall(form_current());

    for (auto& sort : particles_)
      PetscCall(sort->update_cells());
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /// @note Solution is initialized with guess before it is passed into `SNESSolve()`.
  /// The simplest choice is: (E^{n+1/2, k=0}, B^{n+1/2, k=0}) = (E^{n}, B^{n}).
  PetscCall(to_snes(E, B, sol));

  PetscCall(SNESSolve(snes, NULL, sol));
  PetscCall(SNESGetIterationNumber(snes, &snes_iteration_number));

  LOG("SNESSolve() has finished, SNESConvergedReasonView():");
  PetscCall(SNESConvergedReasonView(snes, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(SNESGetSolution(snes, &sol));
  PetscCall(from_snes(sol, E_hk, B_hk));

  PetscCall(VecAXPBY(E, 2, -1, E_hk));
  PetscCall(VecAXPBY(B, 2, -1, B_hk));

  for (auto& sort : particles_) {
    PetscCall(sort->update_cells());
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// SNES callback: advances particles, collects sources, evaluates residuals.
PetscErrorCode Simulation::form_iteration(SNES, Vec vx, Vec vf, void* ctx)
{
  PetscFunctionBeginUser;
  auto* simulation = (Simulation*)ctx;
  PetscCall(simulation->from_snes(vx, simulation->E_hk, simulation->B_hk));
  PetscCall(simulation->form_current());
  PetscCall(simulation->form_function(vf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_current()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(J, 0.0));
  PetscCall(VecSet(M, 0.0));

  for (auto& sort : particles_) {
    PetscCall(VecSet(sort->J, 0.0));
    PetscCall(VecSet(sort->M, 0.0));
    PetscCall(VecSet(sort->J_loc, 0.0));
    PetscCall(VecSet(sort->M_loc, 0.0));
  }

  PetscCall(VecCopy(B_hk, Bn1));
  PetscCall(VecAXPBY(Bn1, -1.0, 2.0, B));

  PetscCall(DMGlobalToLocal(da, E_hk, INSERT_VALUES, E_loc));
  PetscCall(DMGlobalToLocal(da, B_hk, INSERT_VALUES, B_loc));
  PetscCall(DMGlobalToLocal(da, B, INSERT_VALUES, Bn_loc));
  PetscCall(DMGlobalToLocal(da, Bn1, INSERT_VALUES, Bn1_loc));

  PetscCall(DMDAVecGetArrayRead(da, E_loc, &E_arr));
  PetscCall(DMDAVecGetArrayRead(da, Bn_loc, &Bn_arr));
  PetscCall(DMDAVecGetArrayRead(da, B_loc, &B_arr));
  PetscCall(DMDAVecGetArrayRead(da, Bn1_loc, &Bn1_arr));

  Particles* failed_sort = nullptr;
  for (auto& sort : particles_) {
    sort->E_arr = E_arr;
    sort->B_arr = B_arr;
    sort->Bn_arr = Bn_arr;
    sort->Bn1_arr = Bn1_arr;
    PetscCall(sort->form_iteration());
    PetscCall(VecAXPY(J, 1, sort->J));
    PetscCall(VecAXPY(M, 1, sort->M));

    LOG("Push \"{}\": avg it={}, max it={}, retries={}, leaf failures={}, max leaf FRk={: .4e}, max leaf FVhk={: .4e}",
      sort->parameters.sort_name,
      sort->get_average_iteration_number(),
      sort->get_max_iteration_number(),
      sort->get_push_retries(),
      sort->get_push_leaf_failures(),
      sort->get_max_push_leaf_residue_r(),
      sort->get_max_push_leaf_residue_v());

    if (failed_sort == nullptr &&  //
      sort->get_fail_on_terminal_nonconvergence() &&
      sort->get_push_leaf_failures() > 0) {
      failed_sort = sort.get();
      break;
    }
  }

  PetscCall(VecDot(M, B, &a_MB0));

  PetscCall(DMDAVecRestoreArrayRead(da, E_loc, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, B_loc, &B_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, Bn_loc, &Bn_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, Bn1_loc, &Bn1_arr));

  PetscCheck(failed_sort == nullptr, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED,
    "DK Push terminal non-convergence for sort %s: leaf_failures=%d, max FRk=%g, max FVhk=%g",
    failed_sort->parameters.sort_name.c_str(),
    failed_sort->get_push_leaf_failures(),
    failed_sort->get_max_push_leaf_residue_r(),
    failed_sort->get_max_push_leaf_residue_v());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_function(Vec vf)
{
  PetscFunctionBeginUser;
  Vec E_f;
  PetscCall(DMGetGlobalVector(da, &E_f));

  Vec B_f;
  PetscCall(DMGetGlobalVector(da, &B_f));

  // F(E) = (E^{n+1/2,k} - E^{n}) / (dt / 2) + J^{n+1/2,k} - rot(B^{n+1/2,k}) + rot(M^{n+1/2,k}})
  PetscCall(VecAXPBYPCZ(E_f, +2 / dt, -2 / dt, 0, E_hk, E));
  PetscCall(VecAXPY(E_f, +1, J));
  PetscCall(MatMultAdd(rotM, M, E_f, E_f));
  PetscCall(MatMultAdd(rotB, B_hk, E_f, E_f));

  // F(B) = (B^{n+1/2,k} - B^{n}) / (dt / 2) + rot(E^{n+1/2,k})
  PetscCall(VecAXPBYPCZ(B_f, +2 / dt, -2 / dt, 0, B_hk, B));
  PetscCall(MatMultAdd(rotE, E_hk, B_f, B_f));

  PetscCall(to_snes(E_f, B_f, vf));
  PetscCall(DMRestoreGlobalVector(da, &B_f));

  PetscCall(DMRestoreGlobalVector(da, &E_f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::from_snes(Vec v, Vec vE, Vec vB)
{
  PetscFunctionBeginUser;
  const PetscReal**** arr_v;
  PetscCall(DMDAVecGetArrayWrite(da, vE, &E_arr));
  PetscCall(DMDAVecGetArrayWrite(da, vB, &B_arr));
  PetscCall(DMDAVecGetArrayDOFRead(da_EB, v, &arr_v));

#pragma omp parallel for simd
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    PetscInt x = world.start[X] + g % world.size[X];
    PetscInt y = world.start[Y] + (g / world.size[X]) % world.size[Y];
    PetscInt z = world.start[Z] + (g / world.size[X]) / world.size[Y];

    E_arr[z][y][x][X] = arr_v[z][y][x][0];
    E_arr[z][y][x][Y] = arr_v[z][y][x][1];
    E_arr[z][y][x][Z] = arr_v[z][y][x][2];

    B_arr[z][y][x][X] = arr_v[z][y][x][3];
    B_arr[z][y][x][Y] = arr_v[z][y][x][4];
    B_arr[z][y][x][Z] = arr_v[z][y][x][5];
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(da_EB, v, &arr_v));
  PetscCall(DMDAVecRestoreArrayWrite(da, vE, &E_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, vB, &B_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::to_snes(Vec vE, Vec vB, Vec v)
{
  PetscFunctionBeginUser;
  PetscReal**** arr_v;
  PetscCall(DMDAVecGetArrayRead(da, vE, &E_arr));
  PetscCall(DMDAVecGetArrayRead(da, vB, &B_arr));
  PetscCall(DMDAVecGetArrayDOFWrite(da_EB, v, &arr_v));

#pragma omp parallel for simd
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    PetscInt x = world.start[X] + g % world.size[X];
    PetscInt y = world.start[Y] + (g / world.size[X]) % world.size[Y];
    PetscInt z = world.start[Z] + (g / world.size[X]) / world.size[Y];
    arr_v[z][y][x][0] = E_arr[z][y][x][X];
    arr_v[z][y][x][1] = E_arr[z][y][x][Y];
    arr_v[z][y][x][2] = E_arr[z][y][x][Z];

    arr_v[z][y][x][3] = B_arr[z][y][x][X];
    arr_v[z][y][x][4] = B_arr[z][y][x][Y];
    arr_v[z][y][x][5] = B_arr[z][y][x][Z];
  }

  PetscCall(DMDAVecRestoreArrayDOFWrite(da_EB, v, &arr_v));
  PetscCall(DMDAVecRestoreArrayRead(da, vE, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, vB, &B_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace drift_kinetic
