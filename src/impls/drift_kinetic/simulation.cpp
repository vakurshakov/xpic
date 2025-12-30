#include "simulation.h"

#include "src/utils/geometries.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"


namespace drift_kinetic {

static constexpr PetscReal atol = 1e-7;
static constexpr PetscReal rtol = 1e-7;
static constexpr PetscReal stol = 1e-7;
static constexpr PetscReal divtol = PETSC_DETERMINE;
static constexpr PetscInt maxit = 100;
static constexpr PetscInt maxf = PETSC_UNLIMITED;

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(da, &E));
  PetscCall(DMCreateGlobalVector(da, &B));
  PetscCall(DMCreateGlobalVector(da, &B0));
  PetscCall(DMCreateGlobalVector(da, &J));
  PetscCall(DMCreateGlobalVector(da, &M));
  PetscCall(DMCreateGlobalVector(da, &dBdx));
  PetscCall(DMCreateGlobalVector(da, &dBdy));
  PetscCall(DMCreateGlobalVector(da, &dBdz));
  PetscCall(DMCreateGlobalVector(da, &E_hk));
  PetscCall(DMCreateGlobalVector(da, &B_hk));
  PetscCall(DMCreateLocalVector(da, &E_loc));
  PetscCall(DMCreateLocalVector(da, &B_loc));
  PetscCall(DMCreateLocalVector(da, &dBdx_loc));
  PetscCall(DMCreateLocalVector(da, &dBdy_loc));
  PetscCall(DMCreateLocalVector(da, &dBdz_loc));

  PetscCall(DMSetMatrixPreallocateOnly(da, PETSC_FALSE));
  PetscCall(DMSetMatrixPreallocateSkip(da, PETSC_TRUE));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rotE));
  PetscCall(rotor.create_negative(&rotB));
  PetscCall(MatScale(rotB, -1)); /// @see `Simulation::form_function()`

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, REP3_A(world.bounds), world.st,
    REP3_A(Geom_n), REP3_A(world.procs), 6, world.s, REP3_A(world.lg), &da_EB));

  PetscCall(DMSetUp(da_EB));

  PetscCall(DMCreateGlobalVector(da_EB, &sol));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetType(snes, SNESNGMRES));
  PetscCall(SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf));
  PetscCall(SNESSetDivergenceTolerance(snes, divtol));
  PetscCall(SNESSetFunction(snes, NULL, Simulation::form_iteration, this));
  PetscCall(SNESSetFromOptions(snes));

  /// @todo particle setter?..
  // PetscCall(init_particles(*this, particles_));

  energy_cons = std::make_unique<EnergyConservation>(*this);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(interfaces::Simulation::finalize());

  PetscCall(VecDestroy(&E));
  PetscCall(VecDestroy(&B));
  PetscCall(VecDestroy(&B0));
  PetscCall(VecDestroy(&J));
  PetscCall(VecDestroy(&M));
  PetscCall(VecDestroy(&dBdx));
  PetscCall(VecDestroy(&dBdy));
  PetscCall(VecDestroy(&dBdz));
  PetscCall(VecDestroy(&E_hk));
  PetscCall(VecDestroy(&B_hk));
  PetscCall(VecDestroy(&E_loc));
  PetscCall(VecDestroy(&B_loc));
  PetscCall(VecDestroy(&dBdx_loc));
  PetscCall(VecDestroy(&dBdy_loc));
  PetscCall(VecDestroy(&dBdz_loc));

  PetscCall(MatDestroy(&rotE));
  PetscCall(MatDestroy(&rotB));

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

  /// @note Solution is initialized with guess before it is passed into `SNESSolve()`.
  /// The simplest choice is: (E^{n+1/2, k=0}, B^{n+1/2, k=0}) = (E^{n}, B^{n}).
  PetscCall(to_snes(E, B, sol));
  PetscCall(SNESSolve(snes, NULL, sol));

  LOG("  SNESSolve() has finished, SNESConvergedReasonView():");
  PetscCall(SNESConvergedReasonView(snes, PETSC_VIEWER_STDOUT_WORLD));

  // SNESConvergedReason reason;
  // PetscCall(SNESGetConvergedReason(snes, &reason));
  // PetscCheck(reason >= 0, PetscObjectComm((PetscObject)snes), PETSC_ERR_NOT_CONVERGED, "SNESSolve has not converged");

  PetscCall(SNESGetSolution(snes, &sol));
  PetscCall(from_snes(sol, E_hk, B_hk));
  PetscCall(VecAXPBY(E, 2, -1, E_hk));
  PetscCall(VecAXPBY(B, 2, -1, B_hk));

  for (auto& sort : particles_) {
    /// @todo MPI cells update works with standard `Point` :6(
    /// PetscCall(sort->update_cells());
    PetscCall(sort->correct_coordinates());
  }

  PetscCall(energy_cons->diagnose(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_iteration(
  SNES /* snes */, Vec vx, Vec vf, void* ctx)
{
  PetscFunctionBeginUser;
  auto* simulation = (Simulation*)ctx;
  PetscCall(simulation->from_snes(vx, simulation->E_hk, simulation->B_hk));

  /// @todo dBdx, dBdy, dBdz should be computed with `DriftKineticEsirkepov::set_dBidrj()` here

  PetscCall(simulation->form_current());
  PetscCall(simulation->form_function(vf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_current()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(J, 0.0));

  for (auto& sort : particles_) {
    PetscCall(VecSet(sort->J, 0.0));
    PetscCall(VecSet(sort->M, 0.0));
    PetscCall(VecSet(sort->J_loc, 0.0));
    PetscCall(VecSet(sort->M_loc, 0.0));
  }

  PetscCall(DMGlobalToLocal(da, E_hk, INSERT_VALUES, E_loc));
  PetscCall(DMGlobalToLocal(da, B_hk, INSERT_VALUES, B_loc));
  PetscCall(DMDAVecGetArrayRead(da, E_loc, &E_arr));
  PetscCall(DMDAVecGetArrayRead(da, B_loc, &B_arr));

  for (auto& sort : particles_) {
    sort->E_arr = E_arr;
    sort->B_arr = B_arr;
    PetscCall(sort->form_iteration());
    PetscCall(VecAXPY(J, 1, sort->J));
    PetscCall(VecAXPY(M, 1, sort->M));
  }

  PetscCall(DMDAVecRestoreArrayRead(da, E_loc, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, B_loc, &B_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_function(Vec vf)
{
  PetscFunctionBeginUser;
  Vec E_f, B_f;
  PetscCall(DMGetGlobalVector(da, &E_f));
  PetscCall(DMGetGlobalVector(da, &B_f));

  // F(E) = (E^{n+1/2,k} - E^{n}) / (dt / 2) + J^{n+1/2,k} - rot(B^{n+1/2,k} + M^{n+1/2,k}})
  PetscCall(VecAXPBYPCZ(E_f, +2 / dt, -2 / dt, 0, E_hk, E));
  PetscCall(VecAXPY(E_f, +1, J));
  PetscCall(MatMultAdd(rotB, M, E_f, E_f));
  PetscCall(MatMultAdd(rotB, B_hk, E_f, E_f));

  // F(B) = (B^{n+1/2,k} - B^{n}) / (dt / 2) + rot(E^{n+1/2,k})
  PetscCall(VecAXPBYPCZ(B_f, +2 / dt, -2 / dt, 0, B_hk, B));
  PetscCall(MatMultAdd(rotE, E_hk, B_f, B_f));

  PetscCall(to_snes(E_f, B_f, vf));

  PetscCall(DMRestoreGlobalVector(da, &E_f));
  PetscCall(DMRestoreGlobalVector(da, &B_f));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::from_snes(Vec v, Vec vE, Vec vB)
{
  PetscFunctionBeginUser;
  const PetscReal**** arr_v;
  PetscCall(DMDAVecGetArrayDOFWrite(da, vE, &E_arr));
  PetscCall(DMDAVecGetArrayDOFWrite(da, vB, &B_arr));
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
  PetscCall(DMDAVecRestoreArrayDOFWrite(da, E_hk, &E_arr));
  PetscCall(DMDAVecRestoreArrayDOFWrite(da, B_hk, &B_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::to_snes(Vec vE, Vec vB, Vec v)
{
  PetscFunctionBeginUser;
  PetscReal**** arr_v;
  PetscCall(DMDAVecGetArrayDOFRead(da, vE, &E_arr));
  PetscCall(DMDAVecGetArrayDOFRead(da, vB, &B_arr));
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
  PetscCall(DMDAVecRestoreArrayDOFRead(da, E_hk, &E_arr));
  PetscCall(DMDAVecRestoreArrayDOFRead(da, B_hk, &B_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}


EnergyConservation::EnergyConservation(const Simulation& simulation)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/energy_conservation.txt"),
    simulation(simulation)
{
}

PetscErrorCode EnergyConservation::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t == 0) {
    PetscCall(VecNorm(simulation.E, NORM_2, &w_E));
    PetscCall(VecNorm(simulation.B, NORM_2, &w_B));
    PetscCall(VecDot(simulation.M, simulation.B, &a_MB));
    w_E = 0.5 * POW2(w_E);
    w_B = 0.5 * POW2(w_B);
    dF = a_EJ = 0;
  }

  w_E0 = w_E;
  w_B0 = w_B;
  a_MB0 = a_MB;
  PetscCall(TableDiagnostic::diagnose(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(VecNorm(simulation.E, NORM_2, &w_E));
  PetscCall(VecNorm(simulation.B, NORM_2, &w_B));
  PetscCall(VecDot(simulation.E_hk, simulation.J, &a_EJ));
  PetscCall(VecDot(simulation.M, simulation.B, &a_MB));
  w_E = 0.5 * POW2(w_E);
  w_B = 0.5 * POW2(w_B);
  dF = (w_E - w_E0) + (w_B - w_B0) - (a_MB - a_MB0);
  add(13, "dE+dB+dK", "{: .6e}", dF + dt * a_EJ);
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace drift_kinetic
