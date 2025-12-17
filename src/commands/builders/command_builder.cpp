#include "command_builder.h"

#include "src/commands/builders/fields_damping_builder.h"
#include "src/commands/builders/inject_particles_builder.h"
#include "src/commands/builders/remove_particles_builder.h"
#include "src/commands/builders/set_magnetic_field_builder.h"
#include "src/commands/builders/set_particles_builder.h"
#include "src/diagnostics/builders/simulation_backup_builder.h"

CommandBuilder::CommandBuilder(
  interfaces::Simulation& simulation, std::vector<Command_up>& result)
  : Builder(simulation), commands_(result)
{
}

PetscErrorCode build_commands(interfaces::Simulation& simulation,
  std::string_view name, std::vector<Command_up>& result)
{
  PetscFunctionBeginUser;
  using namespace interfaces;

  const Configuration::json_t& config = CONFIG().json;

  if (name == "Presets" && CONFIG().is_loaded_from_backup()) {
    PetscCall(Builder::use_impl<SimulationBackupCommBuilder>(config.at("SimulationBackup"), simulation, result));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  auto&& it = config.find(name);
  if (it == config.end() || it->empty())
    PetscFunctionReturn(PETSC_SUCCESS);

  LOG("Building commands from \"{}\"", name);

  for (auto&& info : *it) {
    if (!info.contains("command"))
      continue;

    std::string command;
    info.at("command").get_to(command);

    if (command == "SetParticles") {
      PetscCall(Builder::use_impl<SetParticlesBuilder>(info, simulation, result));
    }
    else if (command == "InjectParticles") {
      PetscCall(Builder::use_impl<InjectParticlesBuilder>(info, simulation, result));
    }
    else if (command == "RemoveParticles") {
      PetscCall(Builder::use_impl<RemoveParticlesBuilder>(info, simulation, result));
    }
    else if (command == "SetMagneticField") {
      PetscCall(Builder::use_impl<SetMagneticFieldBuilder>(info, simulation, result));
    }
    else if (command == "FieldsDamping") {
      PetscCall(Builder::use_impl<FieldsDampingBuilder>(info, simulation, result));
    }
    else {
      throw std::runtime_error("Unknown command name " + command);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
