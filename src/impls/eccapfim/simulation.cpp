#include "simulation.h"

#include "src/diagnostics/charge_conservation.h"
#include "src/diagnostics/energy_conservation.h"
#include "src/utils/geometries.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"


namespace eccapfim {

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  SyncClock init_clock;
  PetscCall(init_clock.push(__FUNCTION__));
  PetscCall(init_vectors());
  PetscCall(init_matrices());
  PetscCall(init_snes_solver());

  PetscCall(init_particles(*this, particles_));

  std::vector<Vec> currents;
  std::vector<const interfaces::Particles*> sorts;
  for (const auto& sort : particles_) {
    currents.emplace_back(sort->global_J);
    sorts.emplace_back(sort.get());
  }
  currents.emplace_back(J);

  auto&& f_diag = std::make_unique<FieldsEnergy>(E, B);
  auto&& p_diag = std::make_unique<ParticlesEnergy>(sorts);

  diagnostics_.emplace_back(std::make_unique<EnergyConservation>(
    *this, std::move(f_diag), std::move(p_diag)));

  diagnostics_.emplace_back(
    std::make_unique<ChargeConservation>(world.da, currents, sorts));

  PetscCall(init_clock.pop());
  LOG("Initialization took {:6.4e} seconds", init_clock.get(__FUNCTION__));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::timestep_implementation(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  PetscCall(init_iteration());
  PetscCall(calc_iteration());
  PetscCall(after_iteration());
  PetscCall(clock.log_timings());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_iteration()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));

  for (auto& sort : particles_)
    PetscCall(sort->prepare_storage());

  /// @note Solution is initialized with guess before it is passed into `SNESSolve()`.
  /// The simplest choice is: (E^{n+1/2, k=0}, B^{n+1/2, k=0}) = (E^{n}, B^{n}).
  PetscCall(to_snes(E, B, sol));

  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::calc_iteration()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));

  PetscCall(SNESSolve(snes, nullptr, sol));

  // Convergence analysis
  const char* name;
  PetscCall(PetscObjectGetName((PetscObject)snes, &name));
  LOG("  SNESSolve() has finished for \"{}\", SNESConvergedReasonView():", name);
  PetscCall(SNESConvergedReasonView(snes, PETSC_VIEWER_STDOUT_WORLD));

  PetscInt len;
  PetscCall(SNESGetConvergenceHistory(snes, nullptr, nullptr, &len));
  LOG("  Convergence history for this solution:");

  for (PetscInt i = 0; i < len; ++i) {
    LOG("    {:2d} SNES Function norm {:e}", i, conv_hist[i]);
  }

  for (const auto& sort : particles_) {
    LOG("  Average number of Crank-Nicolson iterations for \"{}\" is {}",
      sort->parameters.sort_name, sort->get_average_iteration_number());
  }

  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::after_iteration()
{
  PetscFunctionBeginUser;
  PetscCall(SNESGetSolution(snes, &sol));
  PetscCall(from_snes(sol, E_hk, B_hk));
  PetscCall(VecAXPBY(E, 2, -1, E_hk));
  PetscCall(VecAXPBY(B, 2, -1, B_hk));

  for (auto& sort : particles_)
    PetscCall(sort->update_cells());
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::form_iteration(
  SNES /* snes */, Vec vx, Vec vf, void* ctx)
{
  PetscFunctionBeginUser;
  auto* simulation = (Simulation*)ctx;

  /// @note The clock will show the time of only the last iteration
  PetscCall(simulation->clock.push(__FUNCTION__));

  PetscCall(simulation->from_snes(vx, simulation->E_hk, simulation->B_hk));
  PetscCall(simulation->clear_sources());
  PetscCall(simulation->form_current());
  PetscCall(simulation->form_function(vf));

  PetscCall(simulation->clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(J, 0.0));

  for (auto& sort : particles_)
    PetscCall(sort->clear_sources());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_current()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));

  DM da = world.da;
  PetscCall(DMGlobalToLocal(da, E_hk, INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, B_hk, INSERT_VALUES, local_B));

  Vector3R*** arr_E;
  Vector3R*** arr_B;
  PetscCall(DMDAVecGetArrayRead(da, local_E, &arr_E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &arr_B));

  for (auto& sort : particles_) {
    sort->E = arr_E;
    sort->B = arr_B;
    PetscCall(sort->form_iteration());
    PetscCall(VecAXPY(J, 1.0, sort->global_J));
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &arr_E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &arr_B));

  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_function(Vec vf)
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));

  DM da = world.da;

  Vec E_f, B_f;
  PetscCall(DMGetGlobalVector(da, &E_f));
  PetscCall(DMGetGlobalVector(da, &B_f));

  // F(E) = (E^{n+1/2,k} - E^{n}) / (dt / 2) + J^{n+1/2,k} - rot(B^{n+1/2,k})
  PetscCall(VecAXPBYPCZ(E_f, +2 / dt, -2 / dt, 0, E_hk, E));
  PetscCall(VecAXPY(E_f, +1, J));
  PetscCall(MatMultAdd(rotB, B_hk, E_f, E_f));

  // F(B) = (B^{n+1/2,k} - B^{n}) / (dt / 2) + rot(E^{n+1/2,k})
  PetscCall(VecAXPBYPCZ(B_f, +2 / dt, -2 / dt, 0, B_hk, B));
  PetscCall(MatMultAdd(rotE, E_hk, B_f, B_f));

  PetscCall(to_snes(E_f, B_f, vf));

  PetscCall(DMRestoreGlobalVector(da, &E_f));
  PetscCall(DMRestoreGlobalVector(da, &B_f));

  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}


/// @todo Replace with `DMCreateSubDM()`, `VecGetSubVector()` to obtain E^{n+1/2}, B^{n+1/2}
PetscErrorCode Simulation::from_snes(Vec v, Vec vE, Vec vB)
{
  PetscFunctionBeginUser;
  DM da = world.da;
  PetscReal**** arr_E;
  PetscReal**** arr_B;
  const PetscReal**** arr_v;
  PetscCall(DMDAVecGetArrayDOFWrite(da, vE, &arr_E));
  PetscCall(DMDAVecGetArrayDOFWrite(da, vB, &arr_B));
  PetscCall(DMDAVecGetArrayDOFRead(da_EB, v, &arr_v));

  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    PetscInt x = world.start[X] + g % world.size[X];
    PetscInt y = world.start[Y] + (g / world.size[X]) % world.size[Y];
    PetscInt z = world.start[Z] + (g / world.size[X]) / world.size[Y];

    arr_E[z][y][x][X] = arr_v[z][y][x][0];
    arr_E[z][y][x][Y] = arr_v[z][y][x][1];
    arr_E[z][y][x][Z] = arr_v[z][y][x][2];

    arr_B[z][y][x][X] = arr_v[z][y][x][3];
    arr_B[z][y][x][Y] = arr_v[z][y][x][4];
    arr_B[z][y][x][Z] = arr_v[z][y][x][5];
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(da_EB, v, &arr_v));
  PetscCall(DMDAVecRestoreArrayDOFWrite(da, E_hk, &arr_E));
  PetscCall(DMDAVecRestoreArrayDOFWrite(da, B_hk, &arr_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::to_snes(Vec vE, Vec vB, Vec v)
{
  PetscFunctionBeginUser;
  DM da = world.da;
  const PetscReal**** arr_E;
  const PetscReal**** arr_B;
  PetscReal**** arr_v;
  PetscCall(DMDAVecGetArrayDOFRead(da, vE, &arr_E));
  PetscCall(DMDAVecGetArrayDOFRead(da, vB, &arr_B));
  PetscCall(DMDAVecGetArrayDOFWrite(da_EB, v, &arr_v));

  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    PetscInt x = world.start[X] + g % world.size[X];
    PetscInt y = world.start[Y] + (g / world.size[X]) % world.size[Y];
    PetscInt z = world.start[Z] + (g / world.size[X]) / world.size[Y];

    arr_v[z][y][x][0] = arr_E[z][y][x][X];
    arr_v[z][y][x][1] = arr_E[z][y][x][Y];
    arr_v[z][y][x][2] = arr_E[z][y][x][Z];

    arr_v[z][y][x][3] = arr_B[z][y][x][X];
    arr_v[z][y][x][4] = arr_B[z][y][x][Y];
    arr_v[z][y][x][5] = arr_B[z][y][x][Z];
  }

  PetscCall(DMDAVecRestoreArrayDOFWrite(da_EB, v, &arr_v));
  PetscCall(DMDAVecRestoreArrayDOFRead(da, E_hk, &arr_E));
  PetscCall(DMDAVecRestoreArrayDOFRead(da, B_hk, &arr_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::init_vectors()
{
  PetscFunctionBeginUser;
  DM da = world.da;
  PetscCall(DMCreateGlobalVector(da, &E));
  PetscCall(DMCreateGlobalVector(da, &B));
  PetscCall(DMCreateGlobalVector(da, &J));

  PetscCall(DMCreateGlobalVector(da, &B0));
  PetscCall(DMCreateGlobalVector(da, &E_hk));
  PetscCall(DMCreateGlobalVector(da, &B_hk));
  PetscCall(DMCreateLocalVector(da, &local_E));
  PetscCall(DMCreateLocalVector(da, &local_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_matrices()
{
  PetscFunctionBeginUser;
  DM da = world.da;
  PetscCall(DMSetMatrixPreallocateOnly(da, PETSC_FALSE));
  PetscCall(DMSetMatrixPreallocateSkip(da, PETSC_TRUE));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rotE));
  PetscCall(rotor.create_negative(&rotB));

  /// @note The minus sign here is from Faraday's equation
  PetscCall(MatScale(rotE, -1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_snes_solver()
{
  PetscFunctionBeginUser;
  PetscInt gn[3];
  PetscInt procs[3];
  PetscInt s;
  DMBoundaryType bounds[3];
  DMDAStencilType st;
  PetscCall(DMDAGetInfo(world.da, nullptr, REP3_A(&gn), REP3_A(&procs), nullptr, &s, REP3_A(&bounds), &st));

  const PetscInt* lgn[3];
  PetscCall(DMDAGetOwnershipRanges(world.da, REP3_A(&lgn)));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, REP3_A(bounds), st, REP3_A(gn), REP3_A(procs), 6, st, REP3_A(lgn), &da_EB));
  PetscCall(DMSetUp(da_EB));

  PetscCall(DMCreateGlobalVector(da_EB, &sol));

  static constexpr PetscReal atol = 1e-7;
  static constexpr PetscReal rtol = 1e-7;
  static constexpr PetscReal stol = 1e-7;
  static constexpr PetscReal divtol = 1e+1;
  static constexpr PetscInt maxit = 50;
  static constexpr PetscInt maxf = PETSC_UNLIMITED;

  conv_hist.resize(maxit);

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetType(snes, SNESNGMRES));
  PetscCall(SNESSetErrorIfNotConverged(snes, PETSC_TRUE));
  PetscCall(SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf));
  PetscCall(SNESSetDivergenceTolerance(snes, divtol));
  PetscCall(SNESSetConvergenceHistory(snes, conv_hist.data(), nullptr, maxit, PETSC_TRUE));
  PetscCall(SNESSetFunction(snes, nullptr, Simulation::form_iteration, this));
  PetscCall(SNESSetFromOptions(snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(interfaces::Simulation::finalize());

  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da_EB));
  PetscCall(VecDestroy(&sol));

  PetscCall(MatDestroy(&rotE));
  PetscCall(MatDestroy(&rotB));

  PetscCall(VecDestroy(&E));
  PetscCall(VecDestroy(&B));
  PetscCall(VecDestroy(&J));
  PetscCall(VecDestroy(&B0));
  PetscCall(VecDestroy(&E_hk));
  PetscCall(VecDestroy(&B_hk));
  PetscCall(VecDestroy(&local_E));
  PetscCall(VecDestroy(&local_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}


Vec Simulation::get_named_vector(std::string_view name) const
{
  static const std::unordered_map<std::string_view, Vec> map{
    {"E", E},
    {"B", B},
    {"J", J},
    {"B0", B0},
  };
  return map.at(name);
}

Simulation::NamedValues<Vec> Simulation::get_backup_fields() const
{
  return {{"E", E}, {"B", B}, {"B0", B0}};
}

}  // namespace eccapfim
