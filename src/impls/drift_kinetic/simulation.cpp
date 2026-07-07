#include "simulation.h"

#include <random>

#include "src/algorithms/implicit_drift_kinetic.h"
#include "src/utils/configuration.h"
#include "src/utils/geometries.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"


namespace drift_kinetic {

static constexpr PetscReal atol = 1e-14;
static constexpr PetscReal rtol = 1e-14;
static constexpr PetscReal stol = 0;
static constexpr PetscReal divtol = PETSC_DETERMINE;
static constexpr PetscInt maxit = 1000;
static constexpr PetscInt maxf = PETSC_UNLIMITED;

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(da, &E));
  PetscCall(DMCreateGlobalVector(da, &B));
  PetscCall(DMCreateGlobalVector(da, &B0));
  PetscCall(DMCreateGlobalVector(da, &J));
  PetscCall(DMCreateGlobalVector(da, &M));
  PetscCall(DMCreateGlobalVector(da, &E_hk));
  PetscCall(DMCreateGlobalVector(da, &B_hk));
  PetscCall(DMCreateGlobalVector(da, &Bn1));

  PetscCall(DMCreateLocalVector(da, &E_loc));
  PetscCall(DMCreateLocalVector(da, &B_loc));
  PetscCall(DMCreateLocalVector(da, &Bn_loc));
  PetscCall(DMCreateLocalVector(da, &Bn1_loc));

  PetscCall(DMSetMatrixPreallocateOnly(da, PETSC_FALSE));
  PetscCall(DMSetMatrixPreallocateSkip(da, PETSC_TRUE));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rotE));
  PetscCall(rotor.create_negative(&rotM));
  PetscCall(rotor.create_negative(&rotB));
  PetscCall(MatScale(rotB, -1)); /// @see `Simulation::form_function()`

  PetscInt gn[3];
  PetscInt procs[3];
  PetscInt s;
  DMBoundaryType bounds[3];
  DMDAStencilType st;
  PetscCall(DMDAGetInfo(da, NULL, REP3_A(&gn), REP3_A(&procs), NULL, &s, REP3_A(&bounds), &st));

  const PetscInt* lgn[3];
  PetscCall(DMDAGetOwnershipRanges(da, REP3_A(&lgn)));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, REP3_A(bounds), st, REP3_A(gn), REP3_A(procs), 6, st, REP3_A(lgn), &da_EB));
  PetscCall(DMSetUp(da_EB));

  PetscCall(DMCreateGlobalVector(da_EB, &sol));

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));

  PetscCall(SNESSetType(snes, SNESNGMRES));
  //PetscCall(SNESSetType(snes, SNESANDERSON));

  PetscCall(SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf));
  PetscCall(SNESSetDivergenceTolerance(snes, divtol));
  PetscCall(SNESSetFunction(snes, NULL, Simulation::form_iteration, this));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(init_particles(*this, particles_));

  // Re-walk the JSON to set DK-specific per-sort flags without touching
  // the generic `init_particles` path. `coord_is_gc=true` instructs DK
  // loaders to treat sampled `r` as the guiding center (skip the Larmor
  // shift). This is what allows e/i to coincide at the same GC when both
  // sorts are seeded from a shared coordinate stream.
  {
    const Configuration::json_t& json = CONFIG().json;
    auto it = json.find("Particles");
    if (it != json.end()) {
      for (auto&& info : *it) {
        if (!info.contains("sort_name") || !info.contains("coord_is_gc"))
          continue;
        bool v = false;
        info.at("coord_is_gc").get_to(v);
        auto&& name = info.at("sort_name").get<std::string>();
        for (auto& sort : particles_) {
          if (sort->parameters.sort_name == name) {
            sort->set_coord_is_gc(v);
            LOG("  drift_kinetic: \"{}\" coord_is_gc = {}", name, v);
            break;
          }
        }
      }
    }
  }

  energy_cons = std::make_unique<EnergyConservation>(*this);

  if (!particles_.empty()) {
    trace = std::make_unique<PointByFieldTrace>(CONFIG().out_dir, *particles_[0], 1);
    PetscCall(trace->initialize());
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(energy_cons->diagnose(geom_nt));
  PetscCall(energy_cons->finalize());

  if (trace) {
    PetscCall(trace->diagnose(geom_nt));
    PetscCall(trace->finalize());
  }

  PetscCall(interfaces::Simulation::finalize());

  PetscCall(VecDestroy(&E));
  PetscCall(VecDestroy(&B));
  PetscCall(VecDestroy(&B0));
  PetscCall(VecDestroy(&J));
  PetscCall(VecDestroy(&M));
  PetscCall(VecDestroy(&E_hk));
  PetscCall(VecDestroy(&B_hk));
  PetscCall(VecDestroy(&Bn1));

  PetscCall(VecDestroy(&E_loc));
  PetscCall(VecDestroy(&B_loc));
  PetscCall(VecDestroy(&Bn_loc));
  PetscCall(VecDestroy(&Bn1_loc));

  PetscCall(MatDestroy(&rotE));
  PetscCall(MatDestroy(&rotB));
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
  PetscCall(energy_cons->diagnose(t - 1));
  if (trace) PetscCall(trace->diagnose(t - 1));

  LOG("to_snes():");
  /// @note Solution is initialized with guess before it is passed into `SNESSolve()`.
  /// The simplest choice is: (E^{n+1/2, k=0}, B^{n+1/2, k=0}) = (E^{n}, B^{n}).

  PetscCall(to_snes(E, B, sol));

  LOG("to_snes() has finished, SNESSolve():");

  PetscCall(SNESSolve(snes, NULL, sol));
  PetscCall(SNESGetIterationNumber(snes, &last_field_itnum));

  LOG("SNESSolve() has finished, SNESConvergedReasonView():");
  PetscCall(SNESConvergedReasonView(snes, PETSC_VIEWER_STDOUT_WORLD));

  // SNESConvergedReason reason;
  // PetscCall(SNESGetConvergedReason(snes, &reason));
  // PetscCheck(reason >= 0, PetscObjectComm((PetscObject)snes), PETSC_ERR_NOT_CONVERGED, "SNESSolve has not converged");

  PetscCall(SNESGetSolution(snes, &sol));
  PetscCall(from_snes(sol, E_hk, B_hk));

  PetscCall(VecAXPBY(E, 2, -1, E_hk));
  PetscCall(VecAXPBY(B, 2, -1, B_hk));

  for (auto& sort : particles_) {
    PetscCall(sort->update_cells());
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_iteration(
  SNES /* snes */, Vec vx, Vec vf, void* ctx)
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

  for (auto& sort : particles_) {
    sort->E_arr = E_arr;
    sort->B_arr = B_arr;
    sort->Bn_arr = Bn_arr;
    sort->Bn1_arr = Bn1_arr;
    PetscCall(sort->form_iteration());
    PetscCall(VecAXPY(J, 1, sort->J));
    PetscCall(VecAXPY(M, 1, sort->M));
    LOG("Avg push it = {}, Max push it = {}", sort->get_average_iteration_number(), sort->get_max_iteration_number());
  }

  PetscCall(VecDot(M, B, &energy_cons->a_MB0));

  PetscCall(DMDAVecRestoreArrayRead(da, E_loc, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, B_loc, &B_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, Bn_loc, &Bn_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, Bn1_loc, &Bn1_arr));
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

  PetscCall(VecAXPY(B, -1, B0));
  PetscCall(VecAXPY(B_hk, -1, B0));

  // F(B) = (B^{n+1/2,k} - B^{n}) / (dt / 2) + rot(E^{n+1/2,k})
  PetscCall(VecAXPBYPCZ(B_f, +2 / dt, -2 / dt, 0, B_hk, B));
  PetscCall(MatMultAdd(rotE, E_hk, B_f, B_f));

  PetscCall(VecAXPY(B, +1, B0));
  PetscCall(VecAXPY(B_hk, +1, B0));

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
