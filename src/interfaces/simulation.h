#ifndef SRC_INTERFACES_SIMULATION_H
#define SRC_INTERFACES_SIMULATION_H

#include "src/pch.h"
#include "src/interfaces/diagnostic.h"
#include "src/interfaces/command.h"

class Simulation {
public:
  Simulation() = default;
  virtual ~Simulation() = default;

  virtual PetscErrorCode initialize();
  PetscErrorCode calculate();

protected:
  virtual PetscErrorCode initialize_implementation();
  virtual PetscErrorCode log_information() const;

  virtual PetscErrorCode timestep_implementation(timestep_t timestep);
  PetscErrorCode diagnose(timestep_t timestep) const;

  timestep_t start_ = 0u;

  using Diagnostic_up = std::unique_ptr<Diagnostic>;
  std::vector<Diagnostic_up> diagnostics_;

  using Command_up = std::unique_ptr<Command>;
  std::list<Command_up> step_presets_;
};

#endif  // SRC_INTERFACES_SIMULATION_H
