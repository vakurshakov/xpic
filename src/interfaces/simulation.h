#ifndef SRC_INTERFACES_SIMULATION_H
#define SRC_INTERFACES_SIMULATION_H

#include "src/pch.h"
#include "src/interfaces/world.h"
#include "src/interfaces/command.h"
#include "src/interfaces/diagnostic.h"

namespace interfaces {

class Simulation {
public:
  Simulation() = default;
  virtual ~Simulation() = default;

  World world_;

  virtual PetscErrorCode initialize();
  PetscErrorCode calculate();

protected:
  /// @todo create common global function with sizes and reuse it
  PetscInt index(PetscInt k, PetscInt j, PetscInt i);
  PetscInt index(PetscInt k, PetscInt j, PetscInt i, PetscInt l);

  virtual PetscErrorCode initialize_implementation() = 0;
  virtual PetscErrorCode log_information() const;

  virtual PetscErrorCode timestep_implementation(timestep_t timestep) = 0;
  PetscErrorCode diagnose(timestep_t timestep) const;

  timestep_t start_ = 0u;

  std::list<Command_up> step_presets_;
  std::vector<Diagnostic_up> diagnostics_;
};

}  // namespace interfaces

using Simulation_up = std::unique_ptr<interfaces::Simulation>;

/// @returns Concrete simulation using `config` specification
Simulation_up build_simulation();

#endif  // SRC_INTERFACES_SIMULATION_H
