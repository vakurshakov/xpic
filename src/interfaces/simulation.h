#ifndef SRC_INTERFACES_SIMULATION_H
#define SRC_INTERFACES_SIMULATION_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/diagnostic.h"
#include "src/interfaces/world.h"

namespace interfaces {

class Simulation {
public:
  DEFAULT_MOVABLE(Simulation);

  Simulation() = default;
  virtual ~Simulation() = default;

  World world_;

  PetscErrorCode initialize();
  PetscErrorCode calculate();

protected:
  virtual PetscErrorCode initialize_implementation() = 0;
  virtual PetscErrorCode log_information() const;

  virtual PetscErrorCode timestep_implementation(timestep_t timestep) = 0;
  PetscErrorCode execute(timestep_t timestep);
  PetscErrorCode diagnose(timestep_t timestep);

  timestep_t start_ = 0;

  std::list<Command_up> step_presets_;
  std::vector<Diagnostic_up> diagnostics_;
};

}  // namespace interfaces

using Simulation_up = std::unique_ptr<interfaces::Simulation>;

/// @returns Concrete simulation using `config` specification
Simulation_up build_simulation();

#endif  // SRC_INTERFACES_SIMULATION_H
