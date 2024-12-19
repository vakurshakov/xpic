#include "setup_magnetic_field_builder.h"

#include "src/commands/setup_magnetic_field.h"

SetupMagneticFieldBuilder::SetupMagneticFieldBuilder(
  const interfaces::Simulation& simulation, std::list<Command_up>& result)
  : CommandBuilder(simulation, result)
{
}

PetscErrorCode SetupMagneticFieldBuilder::build(const Configuration::json_t& json)
{
  PetscFunctionBeginUser;

  std::string field;
  Vector3R value;

  std::string message;
  try {
    json.at("field").get_to(field);
    value = parse_vector(json, "value");
  }
  catch (const std::exception& e) {
    message = e.what();
    message += usage_message();
    throw std::runtime_error(message);
  }

  auto&& diag =
    std::make_unique<SetupMagneticField>(get_global_vector(field), value);

  commands_.emplace_back(std::move(diag));

  LOG("Setup magnetic field command is added for {}", field);
  PetscFunctionReturn(PETSC_SUCCESS);
}
