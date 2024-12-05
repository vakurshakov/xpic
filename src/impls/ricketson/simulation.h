#ifndef SRC_RICKETSON_SIMULATION_H
#define SRC_RICKETSON_SIMULATION_H

#include "src/interfaces/simulation.h"
#include "src/impls/ricketson/particles.h"

namespace ricketson {

class Simulation : public interfaces::Simulation {
public:
  Simulation() = default;
  ~Simulation() override;

  Vec E_;
  Vec B_;
  Vec DB_;
  std::vector<Particles> particles_;

  PetscErrorCode calculate_b_norm_gradient();

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(timestep_t timestep) override;

  PetscErrorCode setup_norm_gradient();

  Vec B_norm_;
  Mat norm_gradient_;
};

}  // namespace ricketson

#endif  // SRC_RICKETSON_SIMULATION_H
