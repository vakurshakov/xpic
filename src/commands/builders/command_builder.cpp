#include "command_builder.h"

CommandBuilder::CommandBuilder(
  const interfaces::Simulation& simulation, std::list<Command_up>& result)
  : Builder(simulation), commands_(result)
{
}

/* static */ PetscErrorCode build_commands(
  const interfaces::Simulation& /* simulation */, std::string_view /* name */,
  std::list<Command_up>& /* result */)
{
  return PETSC_SUCCESS;
}
