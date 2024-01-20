#ifndef SRC_IMPLEMENTATIONS_BASIC_SIMULATION_H
#define SRC_IMPLEMENTATIONS_BASIC_SIMULATION_H

#include "src/interfaces/simulation.h"

namespace basic {

class Simulation : public interfaces::Simulation {

  virtual PetscErrorCode initialize_implementation() {
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  virtual PetscErrorCode log_information() const {
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  virtual PetscErrorCode timestep_implementation(timestep_t timestep) {
    PetscFunctionReturn(PETSC_SUCCESS);
  }

};

}

#endif // SRC_IMPLEMENTATIONS_BASIC_SIMULATION_H
