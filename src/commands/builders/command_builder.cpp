#include "command_builder.h"

#include "src/impls/basic/builders/distribution_moment_builder.h"
#include "src/impls/basic/builders/field_view_builder.h"
#include "src/impls/basic/builders/fields_energy_builder.h"
#include "src/utils/region_operations.h"


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
