#include "set_magnetic_field_builder.h"

#include "src/commands/set_magnetic_field.h"

SetMagneticFieldBuilder::SetMagneticFieldBuilder(
  interfaces::Simulation& simulation, std::vector<Command_up>& result)
  : CommandBuilder(simulation, result)
{
}

PetscErrorCode SetMagneticFieldBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  std::set<std::string_view> available_setters{
    "SetUniformField",
    "SetCoilsField",
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
    LOG("  Using SetUniformField setter");
    Vector3R value = parse_vector(setter, "value");
    setup = SetUniformField(value);
    LOG("    Magnetic field value: {} {} {}", REP3_A(value));
  }
  else if (name == "SetCoilsField") {
    LOG("  Using SetCoilsField setter");
    std::vector<SetCoilsField::Coil> coils;
    for (auto& coil_info : setter.at("coils")) {
      SetCoilsField::Coil coil{
        coil_info.at("z0").get<PetscReal>(),
        coil_info.at("R").get<PetscReal>(),
        coil_info.at("I").get<PetscReal>(),
      };
      LOG("    Adding magnetic coil, z0: {}, R: {}, I: {}", coil.z0, coil.R, coil.I);
      coils.emplace_back(std::move(coil));
    }
    setup = SetCoilsField(std::move(coils));
  }

  commands_.emplace_back(std::make_unique<SetMagneticField>(
    simulation_.get_named_vector(field), std::move(setup)));

  LOG("  SetMagneticField command is added for {}", field);
  PetscFunctionReturn(PETSC_SUCCESS);
}
