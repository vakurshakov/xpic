#ifndef SRC_BASIC_SIMULATION_H
#define SRC_BASIC_SIMULATION_H

#include "src/interfaces/simulation.h"

#include <petscdmda.h>

#include "src/impls/ricketson/particles.h"

namespace ricketson {

class Simulation : public interfaces::Simulation {
public:
  Simulation() = default;
  ~Simulation() override;

  DM da_;
  DMBoundaryType bounds_[3];
  std::vector<Particles> particles_;

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(timestep_t timestep) override;
};

}

#endif // SRC_BASIC_SIMULATION_H
