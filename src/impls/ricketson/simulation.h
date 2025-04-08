#ifndef SRC_RICKETSON_SIMULATION_H
#define SRC_RICKETSON_SIMULATION_H

#include "src/interfaces/simulation.h"
#include "src/impls/ricketson/particles.h"

namespace ricketson {

class Simulation final : public interfaces::Simulation {
public:
  DEFAULT_MOVABLE(Simulation);

  Simulation() = default;
  PetscErrorCode finalize() override;

  Vec E_;
  Vec B_;
  Vec DB_;
  std::vector<std::shared_ptr<Particles>> particles_;

  PetscErrorCode calculate_b_norm_gradient();

  Vec get_named_vector(std::string_view name) const override;
  NamedValues<Vec> get_backup_fields() const override;

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(PetscInt timestep) override;

  PetscErrorCode setup_norm_gradient();

  Vec B_norm_;
  Mat norm_gradient_;
};

}  // namespace ricketson

#endif  // SRC_RICKETSON_SIMULATION_H
