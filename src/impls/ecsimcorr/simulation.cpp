#include "simulation.h"

#include "src/utils/geometries.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"


namespace ecsimcorr {

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  PetscCall(ecsim::Simulation::initialize_implementation());

  J = currJe;

  diagnostics_.emplace_back(std::make_unique<ecsimcorr::Energy>(*this));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::timestep_implementation(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  PetscCall(clear_sources());
  PetscCall(first_push());
  PetscCall(predict_fields());
  PetscCall(second_push());
  PetscCall(correct_fields());
  PetscCall(final_update());
  PetscCall(clock.log_timings());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push("clear_sources2"));
  PetscCall(PetscLogStagePush(stagenums[1]));

  PetscCall(ecsim::Simulation::clear_sources());
  PetscCall(VecSet(currJe, 0.0));

  for (auto& sort : particles_)
    PetscCall(sort->calculate_energy());

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Solving Maxwell's equations to find predicted field `Ep` = E'^{n+1/2}
PetscErrorCode Simulation::predict_fields()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));
  PetscCall(PetscLogStagePush(stagenums[3]));

  // Storing `matL` to reuse it for ECSIM current calculation later
  Mat matA;
  PetscCall(MatDuplicate(matL, MAT_COPY_VALUES, &matA));
  PetscCall(ecsim::Simulation::advance_fields(matA));
  PetscCall(MatDestroy(&matA));

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Solving Maxwell's equation to find correct `Ec` = E^{n+1/2}, satisfying continuity equation
PetscErrorCode Simulation::correct_fields()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(__FUNCTION__));
  PetscCall(PetscLogStagePush(stagenums[5]));

  PetscCall(advance_fields(correct, currJe, Ec));

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::final_update()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push("final_update2"));
  PetscCall(PetscLogStagePush(stagenums[6]));

  for (auto& sort : particles_)
    PetscCall(sort->final_update());

  Vec util;
  PetscReal norm;
  PetscCall(DMGetGlobalVector(world.da, &util));

  PetscCall(MatMultAdd(matL, Ec, currI, currI));  // currI = currI + matL * E^{n+1/2}
  PetscCall(VecWAXPY(util, -1, currI, currJe));  // util = -currI + currJe
  PetscCall(VecNorm(util, NORM_2, &norm));
  PetscCall(DMRestoreGlobalVector(world.da, &util));
  LOG("  Norm of the difference in ECSIM and Esirkepov currents: {:.7f}", norm);

  PetscCall(VecSwap(Ep, Ec));
  PetscCall(ecsim::Simulation::final_update());

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_particles()
{
  PetscFunctionBeginUser;
  PetscCall(interfaces::Simulation::init_particles(*this, particles_));

  // Sharing ownership with the base `ecsim::Simulation`
  std::copy(particles_.begin(), particles_.end(),
    std::back_inserter(ecsim::Simulation::particles_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_vectors()
{
  PetscFunctionBeginUser;
  DM da = world.da;
  PetscCall(ecsim::Simulation::init_vectors());
  PetscCall(DMCreateGlobalVector(da, &Ec));
  PetscCall(DMCreateGlobalVector(da, &currJe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Both `predict` and `correct` use the same constant `matM` as a preconditioner
PetscErrorCode Simulation::init_ksp_solvers()
{
  PetscFunctionBeginUser;
  const std::map<std::string, KSP&> map{
    {"predict", ksp},
    {"correct", correct},
  };

  for (auto&& [name, ksp] : map) {
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(PetscObjectSetName((PetscObject)ksp, name.c_str()));
    PetscCall(KSPSetOptionsPrefix(ksp, (name + "_").c_str()));

    PetscCall(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE));
    PetscCall(KSPSetReusePreconditioner(ksp, PETSC_TRUE));
    PetscCall(KSPSetTolerances(ksp, ecsim::rtol, ecsim::atol, ecsim::divtol, ecsim::maxit));
    PetscCall(KSPSetFromOptions(ksp));
  }

  PetscCall(KSPSetOperators(correct, matM, matM));
  PetscCall(KSPSetUp(correct));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_log_stages()
{
  PetscFunctionBeginUser;
  PetscCall(ecsim::Simulation::init_log_stages());

  stagenums[0] = ecsim::Simulation::stagenums[0];  // Initialization
  stagenums[1] = ecsim::Simulation::stagenums[1];  // Clear sources
  stagenums[2] = ecsim::Simulation::stagenums[2];  // First push
  stagenums[4] = ecsim::Simulation::stagenums[4];  // Second push
  stagenums[6] = ecsim::Simulation::stagenums[5];  // Final update
  PetscCall(PetscLogStageRegister("Predict field", &stagenums[3]));
  PetscCall(PetscLogStageRegister("Correct fields", &stagenums[5]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(ecsim::Simulation::finalize());
  PetscCall(KSPDestroy(&correct));
  PetscCall(VecDestroy(&Ec));
  PetscCall(VecDestroy(&currJe));
  PetscFunctionReturn(PETSC_SUCCESS);
}


Energy::Energy(const ecsimcorr::Simulation& simulation)
  : ::Energy(simulation)
{
}

PetscErrorCode Energy::fill_energy_cons(PetscInt t)
{
  PetscFunctionBeginUser;
  PetscCall(VecAXPY(simulation.B, -1.0, simulation.B0));
  PetscCall(::Energy::fill_energy_cons(t));

  PetscInt off = 3;
  for (const auto& sort : simulation.particles_) {
    auto* particles = dynamic_cast<ecsimcorr::Particles*>(sort.get());
    auto&& name = sort->parameters.sort_name;
    auto&& cwd = particles->lambda_dK;
    auto&& pwd = particles->pred_dK - dt * particles->pred_w;
    auto&& ldk = particles->corr_dK - dt * particles->corr_w;
    energy_cons.add(13, "CWD_" + name, "{: .6e}", cwd, ++off);
    energy_cons.add(13, "PWD_" + name, "{: .6e}", pwd, ++off);
    energy_cons.add(13, "LdK_" + name, "{: .6e}", ldk, ++off);
    ++off;
  }

  /// @note Esirkepov current finally created electric field, so its work should be used
  PetscReal corr_w = 0.0;
  for (const auto& sort : simulation.particles_) {
    corr_w += dynamic_cast<ecsimcorr::Particles*>(sort.get())->corr_w;
  }

  energy_cons.add(13, "WD", "{: .6e}", dK - dt * corr_w);
  PetscCall(VecAXPY(simulation.B, +1.0, simulation.B0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace ecsimcorr
