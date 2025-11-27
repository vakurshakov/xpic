#ifndef SRC_BASIC_SIMULATION_H
#define SRC_BASIC_SIMULATION_H

#include "src/interfaces/simulation.h"
#include "src/impls/basic/particles.h"

namespace basic {

class Simulation final : public interfaces::Simulation {
public:
  Simulation() = default;
  PetscErrorCode finalize() override;

  Vec J;
  std::vector<std::shared_ptr<Particles>> particles_;

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(PetscInt timestep) override;

  PetscErrorCode push_particles();
  PetscErrorCode push_fields();

  Mat rotE;
  Mat rotB;
};

}  // namespace basic

#endif  // SRC_BASIC_SIMULATION_H
