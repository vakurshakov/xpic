#include "command_builder.h"

#include "src/commands/builders/inject_particles_builder.h"

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
  LOG("Building commands for {}", name);

  using namespace interfaces;

  for (auto&& info : commands) {
    std::string name;
    info.at("command").get_to(name);

    if (name == "InjectParticles") {
      PetscCall(Builder::use_impl<InjectParticlesBuilder>(info, simulation, result));
    }
    else {
      throw std::runtime_error("Unknown diagnostic name " + name);
    }
  }

  /// @todo Check uniqueness of result directories
  PetscFunctionReturn(PETSC_SUCCESS);
}
