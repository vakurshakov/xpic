#include "simulation.h"

#include "src/diagnostics/charge_conservation.h"
#include "src/diagnostics/energy_conservation.h"
#include "src/diagnostics/momentum_conservation.h"
#include "src/impls/eccapfim/convergence_history.h"
#include "src/utils/geometries.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"


namespace eccapfim {

static constexpr PetscReal atol = 1e-7;
static constexpr PetscReal rtol = 1e-7;
static constexpr PetscReal stol = 1e-7;
static constexpr PetscReal divtol = PETSC_DETERMINE;
static constexpr PetscInt maxit = 100;
static constexpr PetscInt maxf = PETSC_UNLIMITED;

static constexpr PetscInt ew_version = 3;
static constexpr PetscReal ew_rtol_0 = 0.8;
static constexpr PetscReal ew_gamma = 0.9;
static constexpr PetscReal ew_alpha = 1.5;

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  PetscCall(init_log_stages());

  SyncClock init_clock;
  PetscCall(init_clock.push(__FUNCTION__));
  PetscCall(PetscLogStagePush(stagenums[0]));

  PetscCall(init_vectors());
  PetscCall(init_matrices());
  PetscCall(init_snes_solver());

  PetscCall(init_particles(*this, particles_));

  diagnostics_.emplace_back(std::make_unique<ConvergenceHistory>(*this));

  PetscCall(PetscLogStagePop());
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
  PetscCall(PetscLogStagePush(stagenums[1]));

  for (auto& sort : particles_)
    PetscCall(sort->prepare_storage());

  /// @note Solution is initialized with guess before it is passed into `SNESSolve()`.
  /// The simplest choice is: (E^{n+1/2, k=0}, B^{n+1/2, k=0}) = (E^{n}, B^{n}).
#if SNES_ITERATE_B
  PetscCall(to_snes(E, B, sol));
#else
  PetscCall(VecCopy(E, sol));
  PetscCall(DMGlobalToLocal(da, B, INSERT_VALUES, B_loc));
#endif

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::calc_iteration()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));
  PetscCall(PetscLogStagePush(stagenums[2]));

  PetscCall(SNESSolve(snes, NULL, sol));

  // Convergence analysis
  const char* name;
  PetscCall(PetscObjectGetName((PetscObject)snes, &name));
  LOG("  SNESSolve() has finished for \"{}\", SNESConvergedReasonView():", name);
  PetscCall(SNESConvergedReasonView(snes, PETSC_VIEWER_STDOUT_WORLD));

  PetscInt len;
  PetscCall(SNESGetConvergenceHistory(snes, NULL, NULL, &len));
  LOG("  Convergence history for this solution:");

  for (PetscInt i = 0; i < len; ++i) {
    LOG("    {:2d} SNES Function norm {:e}", i, conv_hist[i]);
  }

  for (const auto& sort : particles_) {
    LOG("  Averaged particle iterations information for \"{}\":", sort->parameters.sort_name);
    LOG("    Number of Crank-Nicolson iterations is {:3.4f}", sort->get_average_iteration_number());
    LOG("    Number of traversed cells is {:3.4f}", sort->get_average_number_of_traversed_cells());
  }

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());

  SNESConvergedReason reason;
  PetscCall(SNESGetConvergedReason(snes, &reason));
  PetscCheck(reason >= 0, PetscObjectComm((PetscObject)snes), PETSC_ERR_NOT_CONVERGED, "SNESSolve has not converged");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::after_iteration()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));
  PetscCall(PetscLogStagePush(stagenums[3]));

  PetscCall(SNESGetSolution(snes, &sol));

#if SNES_ITERATE_B
  PetscCall(from_snes(sol, E_hk, B_hk));
  PetscCall(VecAXPBY(E, 2, -1, E_hk));
  PetscCall(VecAXPBY(B, 2, -1, B_hk));
#else
  PetscCall(VecAXPBY(E, 2, -1, sol));
  PetscCall(MatMultAdd(rotE, sol, B, B));
#endif

  for (auto& sort : particles_)
    PetscCall(sort->update_cells());

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::form_iteration(
  SNES /* snes */, Vec vx, Vec vf, void* ctx)
{
  PetscFunctionBeginUser;
  auto* simulation = (Simulation*)ctx;

  /// @todo The clock shows the time of only the last iteration
  PetscCall(simulation->clock.push(__FUNCTION__));

#if SNES_ITERATE_B
  PetscCall(simulation->from_snes(vx, simulation->E_hk, simulation->B_hk));
#else
  PetscCall(VecCopy(vx, simulation->E_hk));
#endif

  PetscCall(simulation->clear_sources());
  PetscCall(simulation->form_current());
  PetscCall(simulation->form_function(vf));

  PetscCall(simulation->clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::clear_sources()
{
  PetscFunctionBeginUser;
  PetscLogEventBegin(events[0], J, 0, 0, 0);

  PetscCall(VecSet(J, 0.0));

  for (auto& sort : particles_)
    PetscCall(sort->clear_sources());

  PetscLogEventEnd(events[0], J, 0, 0, 0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_current()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));
  PetscCall(DMGlobalToLocal(da, E_hk, INSERT_VALUES, E_loc));

#if SNES_ITERATE_B
  PetscCall(DMGlobalToLocal(da, B_hk, INSERT_VALUES, B_loc));
#endif

  PetscCall(DMDAVecGetArrayRead(da, E_loc, &E_arr));
  PetscCall(DMDAVecGetArrayRead(da, B_loc, &B_arr));

  PetscLogEventBegin(events[1], E_loc, B_loc, J, 0);

  for (auto& sort : particles_) {
    sort->E_arr = E_arr;
    sort->B_arr = B_arr;
    PetscCall(sort->form_iteration());
    PetscCall(VecAXPY(J, 1, sort->J));
  }

  PetscLogEventEnd(events[1], E_loc, B_loc, J, 0);

  PetscCall(DMDAVecRestoreArrayRead(da, E_loc, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, B_loc, &B_arr));

  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_function(Vec vf)
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));

#if SNES_ITERATE_B
  Vec E_f, B_f;
  PetscCall(DMGetGlobalVector(da, &E_f));
  PetscCall(DMGetGlobalVector(da, &B_f));

  PetscLogEventBegin(events[2], vf, E_f, B_f, J);

  // F(E) = (E^{n+1/2,k} - E^{n}) / (dt / 2) + J^{n+1/2,k} - rot(B^{n+1/2,k})
  PetscCall(VecAXPBYPCZ(E_f, +2 / dt, -2 / dt, 0, E_hk, E));
  PetscCall(VecAXPY(E_f, +1, J));
  PetscCall(MatMultAdd(rotB, B_hk, E_f, E_f));

  // F(B) = (B^{n+1/2,k} - B^{n}) / (dt / 2) + rot(E^{n+1/2,k})
  PetscCall(VecAXPBYPCZ(B_f, +2 / dt, -2 / dt, 0, B_hk, B));
  PetscCall(MatMultAdd(rotE, E_hk, B_f, B_f));

  PetscCall(to_snes(E_f, B_f, vf));

  PetscLogEventEnd(events[2], vf, E_f, B_f, J);

  PetscCall(DMRestoreGlobalVector(da, &E_f));
  PetscCall(DMRestoreGlobalVector(da, &B_f));
#else
  PetscLogEventBegin(events[2], vf, E_hk, B, J);

  PetscCall(VecCopy(E_hk, vf));
  PetscCall(MatMultAdd(matM, E_hk, vf, vf));
  PetscCall(VecAXPY(vf, -1, E));
  PetscCall(VecAXPY(vf, +0.5 * dt, J));
  PetscCall(MatMultAdd(rotB, B, vf, vf));

  PetscLogEventEnd(events[2], vf, E_hk, B, J);
#endif

  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}


#if SNES_ITERATE_B
/// @todo Replace with `DMCreateSubDM()`, `VecGetSubVector()` to obtain E^{n+1/2}, B^{n+1/2}
PetscErrorCode Simulation::from_snes(Vec v, Vec vE, Vec vB)
{
  PetscFunctionBeginUser;
  const PetscReal**** arr_v;
  PetscCall(DMDAVecGetArrayDOFWrite(da, vE, &E_arr));
  PetscCall(DMDAVecGetArrayDOFWrite(da, vB, &B_arr));
  PetscCall(DMDAVecGetArrayDOFRead(da_EB, v, &arr_v));

  PetscLogEventBegin(events[3], v, vE, vB, 0);

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

  PetscLogEventEnd(events[3], v, vE, vB, 0);

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

  PetscLogEventBegin(events[4], vE, vB, v, 0);

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

  PetscLogEventEnd(events[4], vE, vB, v, 0);

  PetscCall(DMDAVecRestoreArrayDOFWrite(da_EB, v, &arr_v));
  PetscCall(DMDAVecRestoreArrayDOFRead(da, E_hk, &E_arr));
  PetscCall(DMDAVecRestoreArrayDOFRead(da, B_hk, &B_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif


PetscErrorCode Simulation::init_vectors()
{
  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(da, &E));
  PetscCall(DMCreateGlobalVector(da, &B));
  PetscCall(DMCreateGlobalVector(da, &J));

  PetscCall(DMCreateGlobalVector(da, &B0));
  PetscCall(DMCreateGlobalVector(da, &E_hk));
  PetscCall(DMCreateLocalVector(da, &E_loc));
  PetscCall(DMCreateLocalVector(da, &B_loc));

#if SNES_ITERATE_B
  PetscCall(DMCreateGlobalVector(da, &B_hk));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_matrices()
{
  PetscFunctionBeginUser;
  PetscCall(DMSetMatrixPreallocateOnly(da, PETSC_FALSE));
  PetscCall(DMSetMatrixPreallocateSkip(da, PETSC_TRUE));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rotE));
  PetscCall(rotor.create_negative(&rotB));

#if !SNES_ITERATE_B
  PetscCall(MatProductCreate(rotB, rotE, nullptr, &matM));
  PetscCall(MatProductSetType(matM, MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(matM));
  PetscCall(MatProductSymbolic(matM));
  PetscCall(MatProductNumeric(matM));

  PetscCall(MatScale(matM, +0.25 * dt * dt));
  PetscCall(MatScale(rotB, -0.5 * dt));
  PetscCall(MatScale(rotE, -dt));
#else
  PetscCall(MatScale(rotB, -1));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_snes_solver()
{
  PetscFunctionBeginUser;
#if SNES_ITERATE_B
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
#else
  PetscCall(DMCreateGlobalVector(da, &sol));
#endif

  conv_hist.resize(maxit);

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetType(snes, SNESNGMRES));
  PetscCall(SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf));
  PetscCall(SNESSetDivergenceTolerance(snes, divtol));
  PetscCall(SNESSetConvergenceHistory(snes, conv_hist.data(), NULL, maxit, PETSC_TRUE));
  PetscCall(SNESSetFunction(snes, NULL, Simulation::form_iteration, this));
  PetscCall(SNESKSPSetUseEW(snes, PETSC_TRUE));
  PetscCall(SNESKSPSetParametersEW(snes, ew_version, ew_rtol_0, ew_rtol_0, ew_gamma, ew_alpha, PETSC_CURRENT, PETSC_CURRENT));
  PetscCall(SNESSetFromOptions(snes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_log_stages()
{
  PetscFunctionBeginUser;
  PetscCall(PetscClassIdRegister("eccapfim::Simulation", &classid));
  PetscCall(PetscLogEventRegister("clear_sources", classid, &events[0]));
  PetscCall(PetscLogEventRegister("form_current", classid, &events[1]));
  PetscCall(PetscLogEventRegister("form_function", classid, &events[2]));
  PetscCall(PetscLogEventRegister("from_snes", classid, &events[3]));
  PetscCall(PetscLogEventRegister("to_snes", classid, &events[4]));

  PetscCall(PetscLogStageRegister("Initialization", &stagenums[0]));
  PetscCall(PetscLogStageRegister("Init iteraction", &stagenums[1]));
  PetscCall(PetscLogStageRegister("Calc iteraction", &stagenums[2]));
  PetscCall(PetscLogStageRegister("After iteration", &stagenums[3]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(interfaces::Simulation::finalize());

  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&sol));

  PetscCall(MatDestroy(&rotE));
  PetscCall(MatDestroy(&rotB));

  PetscCall(VecDestroy(&E));
  PetscCall(VecDestroy(&B));
  PetscCall(VecDestroy(&J));
  PetscCall(VecDestroy(&B0));
  PetscCall(VecDestroy(&E_hk));
  PetscCall(VecDestroy(&E_loc));
  PetscCall(VecDestroy(&B_loc));

#if SNES_ITERATE_B
  PetscCall(DMDestroy(&da_EB));
  PetscCall(VecDestroy(&B_hk));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace eccapfim
