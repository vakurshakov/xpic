#ifndef SRC_RICKETSON_SIMULATION_H
#define SRC_RICKETSON_SIMULATION_H

#include "src/interfaces/simulation.h"
#include "src/impls/ricketson/particles.h"

namespace ricketson {

class Simulation : public interfaces::Simulation {
public:
  DEFAULT_MOVABLE(Simulation);

  Simulation() = default;
  ~Simulation() override;

  Vec E_;
  Vec B_;
  Vec DB_;
  std::vector<std::unique_ptr<Particles>> particles_;

  PetscErrorCode calculate_b_norm_gradient();

  Vec get_named_vector(std::string_view name) override;
  Particles& get_named_particles(std::string_view name) override;

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(PetscInt timestep) override;

  PetscErrorCode setup_norm_gradient();

  Vec B_norm_;
  Mat norm_gradient_;
};

}  // namespace ricketson

#endif  // SRC_RICKETSON_SIMULATION_H
