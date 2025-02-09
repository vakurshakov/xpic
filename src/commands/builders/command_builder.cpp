#include "command_builder.h"

#include "src/commands/builders/fields_damping_builder.h"
#include "src/commands/builders/inject_particles_builder.h"
#include "src/commands/builders/remove_particles_builder.h"
#include "src/commands/builders/set_particles_builder.h"
#include "src/commands/builders/set_magnetic_field_builder.h"

CommandBuilder::CommandBuilder(
  const interfaces::Simulation& simulation, std::list<Command_up>& result)
  : Builder(simulation), commands_(result)
{
}

/* static */ PetscErrorCode build_commands(
  const interfaces::Simulation& simulation, std::string_view name,
  std::list<Command_up>& result)
{
  const Configuration::json_t& commands = CONFIG().json.at(name);

  if (commands.empty())
    return PETSC_SUCCESS;

  PetscFunctionBeginUser;
  LOG("Building commands from \"{}\"", name);

  using namespace interfaces;

  for (auto&& info : commands) {
    std::string name;
    info.at("command").get_to(name);

    if (name == "SetParticles") {
      PetscCall(Builder::use_impl<SetParticlesBuilder>(info, simulation, result));
    }
    else if (name == "InjectParticles") {
      PetscCall(Builder::use_impl<InjectParticlesBuilder>(info, simulation, result));
    }
    else if (name == "RemoveParticles") {
      PetscCall(Builder::use_impl<RemoveParticlesBuilder>(info, simulation, result));
    }
    else if (name == "SetMagneticField") {
      PetscCall(Builder::use_impl<SetMagneticFieldBuilder>(info, simulation, result));
    }
    else if (name == "FieldsDamping") {
      PetscCall(Builder::use_impl<FieldsDampingBuilder>(info, simulation, result));
    }
    else {
      throw std::runtime_error("Unknown command name " + name);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
