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
  std::set<std::string_view> available_setters{
    "SetUniformField",
  };

  std::string field;
  info.at("field").get_to(field);

  const Configuration::json_t& setter = info.at("setter");

  std::string name;
  setter.at("name").get_to(name);

  if (!available_setters.contains(name))
    throw std::runtime_error("Unknown setter name " + name);

  SetMagneticField::Setter setup;

  if (name == "SetUniformField") {
    Vector3R value = parse_vector(info, "value");
    setup = SetUniformField(value);
  }

  auto&& diag = std::make_unique<SetMagneticField>(
    simulation_.get_named_vector(field), std::move(setup));

  commands_.emplace_back(std::move(diag));

  LOG("SetMagneticField command is added for {}", field);
  PetscFunctionReturn(PETSC_SUCCESS);
}
