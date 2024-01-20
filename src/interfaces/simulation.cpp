#include "simulation.h"

namespace interfaces {

PetscErrorCode Simulation::initialize() {
  PetscFunctionBeginUser;

  PetscCall(initialize_implementation());
  PetscCall(log_information());

  PetscCall(diagnose(start_));
  LOG_FLUSH();

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::calculate() {
  PetscFunctionBeginUser;
  const Configuration& config = CONFIG();

  for (timestep_t t = start_ + 1; t <= config.time; ++t) {
    LOG_TRACE("timestep: [dt] {},\t[1/w_pe] {:.3f}", t, (t * config.dt));

    for (const Command_up& command : step_presets_) {
      PetscCall(command->execute(t));
    }

    PetscCall(timestep_implementation(t));
    PetscCall(diagnose(t));

    step_presets_.remove_if([t](const Command_up& command) {
      return command->needs_to_be_removed(t);
    });
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::diagnose(timestep_t timestep) const {
  PetscFunctionBegin;

  for (const Diagnostic_up& diagnostic : diagnostics_) {
    PetscCall(diagnostic->diagnose(timestep));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

}
