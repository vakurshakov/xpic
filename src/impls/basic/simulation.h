#ifndef SRC_BASIC_SIMULATION_H
#define SRC_BASIC_SIMULATION_H

#include "src/interfaces/simulation.h"
#include "src/impls/basic/particles.h"

namespace basic {

class Simulation final : public interfaces::Simulation {
public:
  DEFAULT_MOVABLE(Simulation);

  Simulation() = default;
  PetscErrorCode finalize() override;

  Vec E;
  Vec B;
  Vec J;
  std::vector<std::shared_ptr<Particles>> particles_;

  Vec get_named_vector(std::string_view name) const override;
  NamedValues<Vec> get_backup_fields() const override;

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(PetscInt timestep) override;

  PetscErrorCode push_particles();
  PetscErrorCode push_fields();

  Vec B0;
  Vec local_E;
  Vec local_B;

  Mat rotE;
  Mat rotB;
};

}  // namespace basic

#endif  // SRC_BASIC_SIMULATION_H
