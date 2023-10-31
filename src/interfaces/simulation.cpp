#include "simulation.h"

void Simulation::initialize() {
  log_information();
  diagnose(start_);
  LOG_FLUSH();
}

void Simulation::calculate() {
  for (timestep_t t = start_ + 1; t <= TIME; ++t) {
    LOG_TRACE("one timestep");
    PROFILE_SCOPE("one timestep");

    for (auto& command : step_presets_) {
      command->execute(t);
    }

    timestep_implementation(t);
    diagnose(t);

    step_presets_.remove_if([t](const Command_up& command) {
      return command->needs_to_be_removed(t);
    });
  }
}

void Simulation::diagnose(timestep_t timestep) const {
  PROFILE_FUNCTION();

  #pragma omp parallel for if(timestep % DIAGNOSE_PERIOD == 0)
  for (const auto& diagnostic : diagnostics_) {
    diagnostic->diagnose(timestep);
  }
}

void Simulation::log_information() const {}

void Simulation::timestep_implementation(timestep_t timestep) {}

void Simulation::information() const {}
