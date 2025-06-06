#ifndef SRC_ECSIMCORR_SIMULATION_H
#define SRC_ECSIMCORR_SIMULATION_H

#include "src/impls/ecsim/simulation.h"
#include "src/impls/ecsimcorr/particles.h"
#include "src/utils/sync_clock.h"

namespace ecsimcorr {

class Simulation final : public ecsim::Simulation {
public:
  Simulation() = default;
  PetscErrorCode finalize() override;

  using ecsim::Simulation::B;
  using ecsim::Simulation::B0;
  using ecsim::Simulation::currI;
  using ecsim::Simulation::E;
  using ecsim::Simulation::Ep;
  using ecsim::Simulation::matL;

  Vec Ec;
  Vec currJe;

  std::vector<std::shared_ptr<Particles>> particles_;

  Vec get_named_vector(std::string_view name) const override;

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(PetscInt t) override;

  PetscErrorCode init_particles() override;
  PetscErrorCode init_vectors() override;
  PetscErrorCode init_ksp_solvers() override;
  PetscErrorCode init_log_stages() override;

  // The main simulation steps
  PetscErrorCode clear_sources();
  PetscErrorCode predict_fields();
  PetscErrorCode correct_fields();
  PetscErrorCode final_update();

  using ecsim::Simulation::ksp; // prediction KSP
  KSP correct;

  PetscLogStage stagenums[7];

  friend class EnergyConservation;
};

}  // namespace ecsimcorr

#endif  // SRC_ECSIMCORR_SIMULATION_H
