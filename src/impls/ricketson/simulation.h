#ifndef SRC_RICKETSON_SIMULATION_H
#define SRC_RICKETSON_SIMULATION_H

#include "src/interfaces/simulation.h"

#include <petscdmda.h>

#include "src/impls/ricketson/particles.h"

namespace ricketson {

class Simulation : public interfaces::Simulation {
public:
  Simulation() = default;
  ~Simulation() override;

  DM da_;
  Vec E_;
  Vec B_;
  Vec B_grad_;
  Vec B_norm_;

  DMBoundaryType bounds_[3];
  std::vector<Particles> particles_;

  PetscErrorCode calculate_B_norm_gradient();

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(timestep_t timestep) override;

  PetscErrorCode setup_norm_gradient();
  PetscInt index(PetscInt k, PetscInt j, PetscInt i);
  PetscInt index(PetscInt k, PetscInt j, PetscInt i, PetscInt l);

  Mat norm_gradient_;
};

}

#endif // SRC_RICKETSON_SIMULATION_H
