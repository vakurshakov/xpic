#ifndef SRC_IMPLEMENTATIONS_BASIC_SIMULATION_H
#define SRC_IMPLEMENTATIONS_BASIC_SIMULATION_H

#include "src/interfaces/simulation.h"

#include <petscdm.h>
#include <petscvec.h>
#include <petscmat.h>

namespace basic {

/// @todo move it into common utils
#define R3C(expr) (expr), (expr), (expr)
#define R3S(expr) (expr); (expr); (expr)
#define R3CX(expr) (expr##x), (expr##y), (expr##z)
#define R3SX(expr) (expr##x); (expr##y); (expr##z)

class Simulation : public interfaces::Simulation {
public:
  Simulation() = default;
  ~Simulation() override;

protected:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(timestep_t timestep) override;

private:
  PetscErrorCode setup_gradient();
  PetscErrorCode setup_rotor();

  DM da_;
  Vec E_;
  Vec B_;
  Mat gradient_;
  Mat rotor_;
};

}

#endif // SRC_IMPLEMENTATIONS_BASIC_SIMULATION_H
