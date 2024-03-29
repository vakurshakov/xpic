#ifndef SRC_INTERFACES_SIMULATION_H
#define SRC_INTERFACES_SIMULATION_H

#include "src/pch.h"
#include "src/interfaces/diagnostic.h"
#include "src/interfaces/command.h"

namespace interfaces {

class Simulation {
public:
  virtual ~Simulation() = default;

  virtual PetscErrorCode initialize();
  PetscErrorCode calculate();

protected:
  virtual PetscErrorCode initialize_implementation() = 0;
  virtual PetscErrorCode log_information() const;

  virtual PetscErrorCode timestep_implementation(timestep_t timestep) = 0;
  PetscErrorCode diagnose(timestep_t timestep) const;

  timestep_t start_ = 0u;

  std::vector<Diagnostic_up> diagnostics_;
  std::list<Command_up> step_presets_;
};

}

using Simulation_up = std::unique_ptr<interfaces::Simulation>;

/// @returns Concrete simulation using `config` specification
Simulation_up build_simulation();

#endif  // SRC_INTERFACES_SIMULATION_H
