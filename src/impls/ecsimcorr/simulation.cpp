#include "simulation.h"

#include "src/diagnostics/charge_conservation.h"
#include "src/impls/ecsimcorr/energy_conservation.h"
#include "src/utils/geometries.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"


namespace ecsimcorr {

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  PetscCall(ecsim::Simulation::initialize_implementation());

  SyncClock init_clock;
  PetscCall(init_clock.push(__FUNCTION__));
  PetscCall(init_log_stages());
  PetscCall(PetscLogStagePush(stagenums[0]));

  std::vector<Vec> currents;
  std::vector<const interfaces::Particles*> particles;
  for (const auto& sort : particles_) {
    currents.emplace_back(sort->global_currJe);
    particles.emplace_back(sort.get());
  }
  currents.emplace_back(currJe);

  // clang-format off
  diagnostics_.pop_back();  // EnergyConservation from ecsim
  diagnostics_.emplace_back(std::make_unique<EnergyConservation>(*this));
  diagnostics_.emplace_back(std::make_unique<ChargeConservation>(world.da, currents, particles));
  // clang-format on

  PetscCall(PetscLogStagePop());
  PetscCall(init_clock.pop());
  LOG("Initialization of ecsimcorr took {:6.4e} seconds", init_clock.get(__FUNCTION__));
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

  static constexpr PetscReal atol = 1e-10;
  static constexpr PetscReal rtol = 1e-10;

  for (auto&& [name, ksp] : map) {
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(PetscObjectSetName((PetscObject)ksp, name.c_str()));
    PetscCall(KSPSetOptionsPrefix(ksp, (name + "_").c_str()));

    PetscCall(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE));
    PetscCall(KSPSetReusePreconditioner(ksp, PETSC_TRUE));
    PetscCall(KSPSetTolerances(ksp, rtol, atol, PETSC_DEFAULT, PETSC_DEFAULT));
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

  stagenums[0] = ecsim::Simulation::stagenums[0]; // Initialization
  stagenums[1] = ecsim::Simulation::stagenums[1]; // Clear sources
  stagenums[2] = ecsim::Simulation::stagenums[2]; // First push
  stagenums[4] = ecsim::Simulation::stagenums[4]; // Second push
  stagenums[6] = ecsim::Simulation::stagenums[5]; // Final update
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


Vec Simulation::get_named_vector(std::string_view name) const
{
  static const std::unordered_map<std::string_view, Vec> map{
    {"E", E},
    {"E_pred", Ep},
    {"E_corr", Ec},
    {"B", B},
    {"B0", B0},
    {"J_ecsim", currI},
    {"J_esirkepov", currJe},
  };
  return map.at(name);
}

}  // namespace ecsimcorr
