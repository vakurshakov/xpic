#include "simulation.h"

#include "src/impls/basic/simulation.h"

namespace interfaces {

PetscErrorCode Simulation::initialize() {
  PetscFunctionBeginUser;

  PetscCall(initialize_implementation());
  PetscCall(log_information());

  PetscCall(diagnose(start_));
  LOG_FLUSH();

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::log_information() const {
  PetscFunctionBeginUser;

  static const double n0 = sqrt(1e13);
  LOG_INFO("Note: Dimensionless units are used.");
  LOG_INFO("For reference, using density 1e13 cm^(-3):");
  LOG_INFO("  frequency,   w_pe = {:.2e} [1/sec]", 5.64e+4 * n0);
  LOG_INFO("  time,      1/w_pe = {:.2e} [sec]",   1.77e-5 / n0);
  LOG_INFO("  length,    c/w_pe = {:.2e} [cm]",    5.32e+5 / n0);
  LOG_INFO("  electric field, E = {:.2e} [MV/cm]", 9.63e-7 * n0);
  LOG_INFO("  magnetic field, B = {:.2e} [T]",     3.21e-7 * n0);

  LOG_INFO("Geometric constants for the current setup:");
  LOG_INFO("  (length along x axis) = {:.2e} [c/w_pe] = {} [dx]", geom_x, geom_nx);
  LOG_INFO("  (length along y axis) = {:.2e} [c/w_pe] = {} [dy]", geom_y, geom_ny);
  LOG_INFO("  (length along z axis) = {:.2e} [c/w_pe] = {} [dz]", geom_z, geom_nz);
  LOG_INFO("  (simulation time)     = {:.2e} [1/w_pe] = {} [dt]", geom_t, geom_nt);

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::calculate() {
  PetscFunctionBeginUser;
  for (timestep_t t = start_ + 1; t <= geom_nt; ++t) {
    LOG_TRACE("timestep = {:4.3f} [1/w_pe]\t= {} [dt]", (t * dt), t);

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

Simulation_up build_simulation() {
  Simulation_up simulation = nullptr;

  std::string simulation_str;

  const Configuration& config = CONFIG();
  config.json.at("Simulation").get_to(simulation_str);

  if (simulation_str == "basic") {
    return std::make_unique<basic::Simulation>();
  }

  throw std::runtime_error("Unkown simulation is used: " + simulation_str);
}
