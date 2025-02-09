#include "set_magnetic_field_builder.h"

#include "src/commands/set_magnetic_field.h"

SetMagneticFieldBuilder::SetMagneticFieldBuilder(
  const interfaces::Simulation& simulation, std::list<Command_up>& result)
  : CommandBuilder(simulation, result)
{
}

PetscErrorCode SetMagneticFieldBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;

  std::string field;
  info.at("field").get_to(field);

  Vector3R value = parse_vector(info, "value");

  auto&& diag = std::make_unique<SetMagneticField>(
    simulation_.get_named_vector(field), SetUniformField(value));

  commands_.emplace_back(std::move(diag));

  LOG("SetMagneticField command is added for {}", field);
  PetscFunctionReturn(PETSC_SUCCESS);
}
