#include "setup_magnetic_field_builder.h"

#include "src/commands/setup_magnetic_field.h"

SetupMagneticFieldBuilder::SetupMagneticFieldBuilder(
  const interfaces::Simulation& simulation, std::list<Command_up>& result)
  : CommandBuilder(simulation, result)
{
}

PetscErrorCode SetupMagneticFieldBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;

  std::string field;
  info.at("field").get_to(field);

  Vector3R value = parse_vector(info, "value");

  auto&& diag = std::make_unique<SetupMagneticField>(
    simulation_.get_named_vector(field), value);

  commands_.emplace_back(std::move(diag));

  LOG("SetupMagneticField command is added for {}", field);
  PetscFunctionReturn(PETSC_SUCCESS);
}
