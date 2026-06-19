#include "simulation.h"

#include "src/utils/geometries.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"


namespace eccapfim {

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  PetscCall(init_log());

  SyncClock init_clock;
  PetscCall(init_clock.push(__FUNCTION__));
  PetscCall(PetscLogStagePush(stagenums[0]));

  PetscCall(init_snes());
  PetscCall(init_particles(*this, particles_));

  diagnostics_.emplace_back(std::make_unique<ConvergenceHistory>(*this));

  if (!particles_.empty())
    diagnostics_.emplace_back(
      std::make_unique<ParticleTrace>(CONFIG().out_dir, *particles_[0]));

  PetscCall(PetscLogStagePop());
  PetscCall(init_clock.pop());
  LOG("Initialization took {:6.4e} seconds", init_clock.get(__FUNCTION__));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_snes()
{
  PetscFunctionBeginUser;
  conv_hist.resize(maxit);

  /// @note `SNESANDERSON` can be more effective: -snes_type anderson -snes_anderson_beta 0.4
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetType(snes, SNESNGMRES));
  PetscCall(SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf));
  PetscCall(SNESSetDivergenceTolerance(snes, divtol));
  PetscCall(SNESSetConvergenceHistory(snes, conv_hist.data(), NULL, maxit, PETSC_TRUE));
  PetscCall(SNESSetFunction(snes, NULL, Simulation::form_iteration, this));
  PetscCall(SNESKSPSetUseEW(snes, PETSC_TRUE));
  PetscCall(SNESKSPSetParametersEW(snes, ew_version, ew_rtol_0, ew_rtol_0, ew_gamma, ew_alpha, PETSC_CURRENT, PETSC_CURRENT));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMCreateGlobalVector(da, &E_hk));

#if SNES_ITERATE_B || SNES_PRECONDITIONING
  auto create_dm = [&](PetscInt dim, DM* local_dm) {
    Region region{
      .dim = 4,
      .dof = dim,
      .start = Vector4I(0, 0, 0, 0),
      .size = Vector4I(geom_nx, geom_ny, geom_nz, dim),
    };
    return World::create_local_dm(da, region, PETSC_COMM_WORLD, local_dm);
  };
#endif

#if SNES_ITERATE_B
  PetscCall(create_dm(6, &sda));
  PetscCall(DMCreateGlobalVector(da, &B_hk));
  PetscCall(DMCreateGlobalVector(sda, &sol));
#else
  PetscCall(DMCreateGlobalVector(da, &sol));
#endif

  PetscCall(SNESSetSolution(snes, sol));
  PetscCall(SNESSetUp(snes));

#if SNES_PRECONDITIONING
  PetscCall(SNESGetNPC(snes, &npc));
  PetscCall(SNESSetType(npc, SNESSHELL));
  PetscCall(SNESSetNPCSide(npc, PC_RIGHT));
  PetscCall(SNESShellSetContext(npc, this));
  PetscCall(SNESShellSetSolve(npc, Simulation::form_prediction));

  PetscCall(DMCreateGlobalVector(da, &nsol));
  PetscCall(DMCreateGlobalVector(da, &I));
  PetscCall(DMCreateLocalVector(da, &I_loc));

  PetscCall(SNESSetSolution(npc, nsol));
  PetscCall(SNESSetUp(npc));

  PetscInt M = geom_nx * geom_ny * geom_nz * 3;
  PetscCall(create_dm(9, &nda));
  PetscCall(DMCreateGlobalVector(nda, &nLv));
  PetscCall(DMCreateLocalVector(nda, &nLv_loc));
  PetscCall(MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, M, this, &nLm));
  PetscCall(MatShellSetOperation(nLm, MATOP_MULT, (void(*)(void))Simulation::mat_mult));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetTolerances(ksp, rtol, atol, divtol, maxit));  // Should we use here different tolerances?
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_log()
{
  PetscFunctionBeginUser;
  PetscCall(PetscClassIdRegister("eccapfim::Simulation", &classid));
  PetscCall(PetscLogEventRegister("clear_sources", classid, &events[0]));
  PetscCall(PetscLogEventRegister("form_current", classid, &events[1]));
  PetscCall(PetscLogEventRegister("form_function", classid, &events[2]));
  PetscCall(PetscLogEventRegister("form_precond", classid, &events[3]));
  PetscCall(PetscLogEventRegister("comp_snes", classid, &events[4]));
  PetscCall(PetscLogEventRegister("decomp_snes", classid, &events[5]));
  PetscCall(PetscLogEventRegister("mat_mult", classid, &events[6]));

  PetscCall(PetscLogStageRegister("Initialization", &stagenums[0]));
  PetscCall(PetscLogStageRegister("Init iteration", &stagenums[1]));
  PetscCall(PetscLogStageRegister("Calc iteration", &stagenums[2]));
  PetscCall(PetscLogStageRegister("After iteration", &stagenums[3]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(interfaces::Simulation::finalize());

  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&sol));
  PetscCall(VecDestroy(&E_hk));
#if SNES_ITERATE_B
  PetscCall(VecDestroy(&B_hk));
  PetscCall(DMDestroy(&sda));
#endif
#if SNES_PRECONDITIONING
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&I));
  PetscCall(VecDestroy(&I_loc));
  PetscCall(VecDestroy(&nLv));
  PetscCall(VecDestroy(&nLv_loc));
  PetscCall(MatDestroy(&nLm));
  PetscCall(DMDestroy(&nda));
#endif
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

  // Solution is initialized with guess before it is passed into `SNESSolve()`.
  // The simplest choice is: (E^{n+1/2, k=0}, B^{n+1/2, k=0}) = (E^{n}, B^{n}).
  PetscCall(comp_snes(E, B, sol));

#if !SNES_ITERATE_B
  PetscCall(DMGlobalToLocal(da, B, INSERT_VALUES, B_loc));
  // Seed E_loc with the current field too. For a self-consistent run it is
  // overwritten by the solver/NPC; for a fixed-field (test-particle) run, where
  // `form_function` is disabled and the NPC does not refill E_loc, this is what
  // lets the particle actually feel E (and thus ExB-drift) instead of only
  // gyrating around B.
  PetscCall(DMGlobalToLocal(da, E, INSERT_VALUES, E_loc));
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

  const char* name;
  SNESConvergedReason reason;
  PetscInt nit, lit, len;

  PetscCall(PetscObjectGetName((PetscObject)snes, &name));
  PetscCall(SNESGetConvergedReason(snes, &reason));
  PetscCall(SNESGetIterationNumber(snes, &nit));
  PetscCall(SNESGetLinearSolveIterations(snes, &lit));
  PetscCall(SNESGetConvergenceHistory(snes, NULL, NULL, &len));

  LOG("  SNESSolve has finished for \"{}\"", name);
  LOG("    Reason why solver finished: {}", SNESConvergedReasons[reason]);
  LOG("    Total nonlinear iterations: {}", nit);
  LOG("    Average linear iterations: {}", lit / (PetscReal)nit);
  LOG("    Convergence history for this solution:");

  for (PetscInt i = 0; i < len; ++i) {
    LOG("      {:2d} SNES Function norm {:e}", i, conv_hist[i]);
  }

  for (const auto& sort : particles_) {
    LOG("  Particle iterations information for \"{}\":", sort->parameters.sort_name);
    LOG("    Averaged number of Crank-Nicolson iterations is {:3.4f}", sort->get_average_iteration_number());
    LOG("    Averaged number of traversed cells is {:3.4f}", sort->get_average_number_of_traversed_cells());
    LOG("    Maximum number of traversed cells is {:3d}", sort->get_maximum_number_of_traversed_cells());
  }

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());

  PetscCheck(reason >= 0, PetscObjectComm((PetscObject)snes), PETSC_ERR_NOT_CONVERGED, "SNESSolve has not converged");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* static */ PetscErrorCode Simulation::form_prediction(SNES npc, Vec x)
{
  PetscFunctionBeginUser;
  Simulation* ctx;
  PetscCall(SNESShellGetContext(npc, &ctx));
  PetscCall(ctx->impl_form_prediction(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::impl_form_prediction(Vec x)
{
  PetscFunctionBeginUser;
  PetscCall(decomp_snes(E_hk, B_hk, x));
  PetscCall(DMGlobalToLocal(da, E_hk, INSERT_VALUES, E_loc));
#if SNES_ITERATE_B
  PetscCall(DMGlobalToLocal(da, B_hk, INSERT_VALUES, B_loc));
#endif

  PetscCall(DMDAVecGetArrayRead(da, B_loc, &B_arr));
  PetscCall(DMDAVecGetArrayDOFWrite(da, I_loc, &I_arr));
  PetscCall(DMDAVecGetArrayDOFWrite(nda, nLv_loc, &nLv_arr));

  /// @todo The clock shows the time of only the last iteration
  PetscCall(clock.push(__FUNCTION__));

  for (auto& sort : particles_) {
    sort->B_arr = B_arr;
    PetscCall(sort->form_prediction(I_arr, nLv_arr));
  }

  PetscCall(DMDAVecRestoreArrayRead(da, B_loc, &B_arr));
  PetscCall(DMDAVecRestoreArrayDOFWrite(da, I_loc, &I_arr));
  PetscCall(DMDAVecRestoreArrayDOFWrite(nda, nLv_loc, &nLv_arr));

  PetscCall(DMLocalToGlobal(da, I_loc, ADD_VALUES, I));
  PetscCall(DMLocalToGlobal(nda, nLv_loc, ADD_VALUES, nLv));

  PetscCall(MatMult(rotB, B, rB));
  PetscCall(VecCopy(E, rE));
  PetscCall(VecAXPY(rE, -0.5 * dt, I));
  PetscCall(VecAXPY(rE, +0.5 * dt, rB));

  PetscCall(KSPSetOperators(ksp, nLm, nLm));
  PetscCall(KSPSolve(ksp, rE, E_hk));
  PetscCall(KSPGetSolution(ksp, &E_hk));

  PetscCall(clock.pop());

  // Convergence analysis
  const char* name;
  KSPConvergedReason reason;
  PetscInt lit;

  PetscCall(PetscObjectGetName((PetscObject)ksp, &name));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscCall(KSPGetResidualHistory(ksp, NULL, &lit));

  LOG("  KSPSolve() has finished for \"{}\"", name);
  LOG("    Reason why solver finished: {}", KSPConvergedReasons[reason]);
  LOG("    Total linear iterations: {}", lit);

  for (PetscInt i = 0; i < lit; ++i) {
    LOG("      {:2d} KSP Residual norm {:e}", i, conv_hist[i]);
  }

  PetscCheck(reason >= 0, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "KSPSolve has not converged");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* static */ PetscErrorCode Simulation::form_iteration(
  SNES /* snes */, Vec x, Vec f, void* ctx)
{
  PetscFunctionBeginUser;
  PetscCall(((Simulation*)ctx)->impl_form_iteration(x, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::impl_form_iteration(Vec x, Vec f)
{
  PetscFunctionBeginUser;
#if !SNES_PRECONDITIONING
  PetscCall(decomp_snes(E_hk, B_hk, x));
  PetscCall(DMGlobalToLocal(da, E_hk, INSERT_VALUES, E_loc));
  #if SNES_ITERATE_B
  PetscCall(DMGlobalToLocal(da, B_hk, INSERT_VALUES, B_loc));
  #endif
#endif
  PetscCall(clock.push(__FUNCTION__));
  PetscCall(clear_sources());
  PetscCall(form_current());
  //PetscCall(form_function(f));
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::clear_sources()
{
  PetscFunctionBeginUser;
  PetscLogEventBegin(events[0], J, 0, 0, 0);

  PetscCall(VecSet(J, 0));

  for (auto& sort : particles_) {
    PetscCall(VecSet(sort->J, 0));
    PetscCall(VecSet(sort->J_loc, 0));
  }

  PetscLogEventEnd(events[0], J, 0, 0, 0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_current()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));

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

PetscErrorCode Simulation::form_function(Vec f)
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));
#if SNES_ITERATE_B
  Vec E_f, B_f;
  PetscCall(DMGetGlobalVector(da, &E_f));
  PetscCall(DMGetGlobalVector(da, &B_f));

  PetscLogEventBegin(events[2], f, E_f, B_f, J);

  // F(E) = (E^{n+1/2,k} - E^{n}) + (0.5 * dt) * (J^{n+1/2,k} - rot(B^{n+1/2,k}))
  PetscCall(MatMult(rotB, B_hk, rB));
  PetscCall(VecCopy(E_hk, E_f));
  PetscCall(VecAXPY(E_f, -1, E));
  PetscCall(VecAXPY(E_f, +0.5 * dt, J));
  PetscCall(VecAXPY(E_f, -0.5 * dt, rB));

  // F(B) = (B^{n+1/2,k} - B^{n}) + (0.5 * dt) * rot(E^{n+1/2,k})
  PetscCall(MatMult(rotE, E_hk, rE));
  PetscCall(VecCopy(B_hk, B_f));
  PetscCall(VecAXPY(B_f, -1, B));
  PetscCall(VecAXPY(B_f, +0.5 * dt, rE));

  PetscLogEventEnd(events[2], f, E_f, B_f, J);

  PetscCall(comp_snes(E_f, B_f, f));

  PetscCall(DMRestoreGlobalVector(da, &E_f));
  PetscCall(DMRestoreGlobalVector(da, &B_f));
#else
  PetscLogEventBegin(events[2], f, E_hk, B, J);

  PetscCall(MatMult(rotB, B, rB));
  PetscCall(MatMult(matM, E_hk, f));
  PetscCall(VecAXPY(f, -1, E));
  PetscCall(VecAXPY(f, +0.5 * dt, J));
  PetscCall(VecAXPY(f, -0.5 * dt, rB));

  PetscLogEventEnd(events[2], f, E_hk, B, J);
#endif
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::after_iteration()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));
  PetscCall(PetscLogStagePush(stagenums[3]));

  PetscCall(SNESGetSolution(snes, &sol));
  PetscCall(decomp_snes(E_hk, B_hk, sol));

#if SNES_ITERATE_B
  PetscCall(VecAXPBY(E, 2, -1, E_hk));
  PetscCall(VecAXPBY(B, 2, -1, B_hk));
#else
  PetscCall(VecAXPBY(E, 2, -1, E_hk));
  PetscCall(MatMult(rotE, E_hk, rE));
  PetscCall(VecAXPY(B, -dt, rE));
#endif

  for (auto& sort : particles_)
    PetscCall(sort->update_cells());

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* static */ PetscErrorCode Simulation::mat_mult(Mat mat, Vec x, Vec y)
{
  PetscFunctionBeginUser;
  Simulation* sim;
  PetscCall(MatShellGetContext(mat, &sim));
  PetscCall(sim->impl_mat_mult(x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::impl_mat_mult(Vec x, Vec y)
{
  PetscFunctionBeginUser;
  const PetscReal**** L_arr;
  const PetscReal**** x_arr;
  PetscReal**** y_arr;
  PetscCall(DMDAVecGetArrayDOFRead(nda, nLv, &L_arr));
  PetscCall(DMDAVecGetArrayDOFRead(da, x, &x_arr));
  PetscCall(DMDAVecGetArrayDOFWrite(da, y, &y_arr));

  PetscLogEventBegin(events[6], nLv, x, y, 0);

#pragma omp parallel for simd
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    PetscInt c1, c2, x, y, z;

    x = world.start[X] + g % world.size[X];
    y = world.start[Y] + (g / world.size[X]) % world.size[Y];
    z = world.start[Z] + (g / world.size[X]) / world.size[Y];

    for (c1 = 0; c1 < 3; c1++) {
      for (c2 = 0; c2 < 3; c2++) {
        y_arr[z][y][x][c1] += L_arr[z][y][x][c1 * 3 + c2] * x_arr[z][y][x][c2];
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(nda, nLv, &L_arr));
  PetscCall(DMDAVecRestoreArrayDOFRead(da, x, &x_arr));
  PetscCall(DMDAVecRestoreArrayDOFWrite(da, y, &y_arr));

  PetscLogEventEnd(events[6], nLv, x, y, 0);
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::comp_snes(Vec vE, Vec vB, Vec v)
{
  PetscFunctionBeginUser;
#if SNES_ITERATE_B
  const PetscReal**** vE_arr;
  const PetscReal**** vB_arr;
  PetscReal**** v_arr;
  PetscCall(DMDAVecGetArrayDOFRead(da, vE, &vE_arr));
  PetscCall(DMDAVecGetArrayDOFRead(da, vB, &vB_arr));
  PetscCall(DMDAVecGetArrayDOFWrite(sda, v, &v_arr));

  PetscLogEventBegin(events[4], vE, vB, v, 0);

  #pragma omp parallel for simd
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    PetscInt x = world.start[X] + g % world.size[X];
    PetscInt y = world.start[Y] + (g / world.size[X]) % world.size[Y];
    PetscInt z = world.start[Z] + (g / world.size[X]) / world.size[Y];

    v_arr[z][y][x][0] = vE_arr[z][y][x][X];
    v_arr[z][y][x][1] = vE_arr[z][y][x][Y];
    v_arr[z][y][x][2] = vE_arr[z][y][x][Z];

    v_arr[z][y][x][3] = vB_arr[z][y][x][X];
    v_arr[z][y][x][4] = vB_arr[z][y][x][Y];
    v_arr[z][y][x][5] = vB_arr[z][y][x][Z];
  }

  PetscLogEventEnd(events[4], vE, vB, v, 0);

  PetscCall(DMDAVecRestoreArrayDOFWrite(sda, v, &v_arr));
  PetscCall(DMDAVecRestoreArrayDOFRead(da, vE, &vE_arr));
  PetscCall(DMDAVecRestoreArrayDOFRead(da, vB, &vB_arr));
#else
  PetscCall(VecCopy(vE, v));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::decomp_snes(Vec vE, Vec vB, Vec v)
{
  PetscFunctionBeginUser;
#if SNES_ITERATE_B
  PetscReal**** vE_arr;
  PetscReal**** vB_arr;
  const PetscReal**** v_arr;
  PetscCall(DMDAVecGetArrayDOFWrite(da, vE, &vE_arr));
  PetscCall(DMDAVecGetArrayDOFWrite(da, vB, &vB_arr));
  PetscCall(DMDAVecGetArrayDOFRead(sda, v, &v_arr));

  PetscLogEventBegin(events[5], v, vE, vB, 0);

  #pragma omp parallel for simd
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    PetscInt x = world.start[X] + g % world.size[X];
    PetscInt y = world.start[Y] + (g / world.size[X]) % world.size[Y];
    PetscInt z = world.start[Z] + (g / world.size[X]) / world.size[Y];

    vE_arr[z][y][x][X] = v_arr[z][y][x][0];
    vE_arr[z][y][x][Y] = v_arr[z][y][x][1];
    vE_arr[z][y][x][Z] = v_arr[z][y][x][2];

    vB_arr[z][y][x][X] = v_arr[z][y][x][3];
    vB_arr[z][y][x][Y] = v_arr[z][y][x][4];
    vB_arr[z][y][x][Z] = v_arr[z][y][x][5];
  }

  PetscLogEventEnd(events[5], v, vE, vB, 0);

  PetscCall(DMDAVecRestoreArrayDOFRead(sda, v, &v_arr));
  PetscCall(DMDAVecRestoreArrayDOFWrite(da, vE, &vE_arr));
  PetscCall(DMDAVecRestoreArrayDOFWrite(da, vB, &vB_arr));
#else
  PetscCall(VecCopy(v, vE));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}


ConvergenceHistory::ConvergenceHistory(const Simulation& simulation)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/convergence_history.txt"),
    simulation(simulation)
{
}

PetscErrorCode ConvergenceHistory::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  add(6, "Time", "{:d}", t);

  SNES snes = simulation.snes;

  PetscInt nit, lit, len;
  PetscReal* hist;

  PetscCall(SNESGetIterationNumber(snes, &nit));
  PetscCall(SNESGetLinearSolveIterations(snes, &lit));
  PetscCall(SNESGetConvergenceHistory(snes, &hist, nullptr, &len));

  add(6, "NItNum", "{:d}", nit);
  add(6, "LItNum", "{:d}", lit);

  for (const auto& sort : simulation.particles_) {
    const auto& name = sort->parameters.sort_name;
    auto cn = sort->get_average_iteration_number();
    auto atc = sort->get_average_number_of_traversed_cells();
    auto mtc = sort->get_maximum_number_of_traversed_cells();
    add(8, "AvgCN_" + name, "{:.3f}", cn);
    add(8, "AvgTC_" + name, "{:.3f}", atc);
    add(8, "MaxTC_" + name, "{:3d}", mtc);
  }

  add_separator();

  if (len == 0) {
    add(12, "ConvHist", "{}", "");
  }
  else {
    for (PetscInt i = 0; i < len; ++i)
      add(12, "ConvHist", "{:8.6e}", hist[i]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


ParticleTrace::ParticleTrace(
  const std::string& out_dir, const interfaces::Particles& particles)
  : TableDiagnostic(out_dir + "/temporal/particle_trace.txt"), particles(particles)
{
}

PetscErrorCode ParticleTrace::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  for (const auto& cell : particles.storage) {
    if (cell.empty())
      continue;

    const Point& point = cell.front();
    add(20, "t_[1/wpe]", "{: .9e}", t * dt);
    add(20, "x_[c/wpe]", "{: .9e}", point.x());
    add(20, "y_[c/wpe]", "{: .9e}", point.y());
    add(20, "z_[c/wpe]", "{: .9e}", point.z());
    add(20, "px_[mc]", "{: .9e}", point.px());
    add(20, "py_[mc]", "{: .9e}", point.py());
    add(20, "pz_[mc]", "{: .9e}", point.pz());
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace eccapfim
