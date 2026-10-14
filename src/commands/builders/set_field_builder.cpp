#include "set_field_builder.h"

#include "src/commands/set_field.h"

SetFieldBuilder::SetFieldBuilder(
  interfaces::Simulation& simulation, std::vector<Command_up>& result)
  : CommandBuilder(simulation, result)
{
}

PetscErrorCode SetFieldBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  std::set<std::string_view> available_setters{
    "SetUniformField",
    "SetGradientField",
    "SetAzimuthalField",
    "SetCoilsField",
    "SetCosineField",
    "SetGeneralCosineField",
  };

  std::string field;
  info.at("field").get_to(field);

  Vec F = NULL;
  Vec F0 = NULL;

  if (field == "E") {
    F = simulation_.E;
  }
  else if (field == "B") {
    F = simulation_.B;
    F0 = simulation_.B0;
  }
  else {
    throw std::runtime_error("Unknown field name " + field + ", expected E or B");
  }

  const Configuration::json_t& setter = info.at("setter");

  std::string name;
  setter.at("name").get_to(name);

  if (!available_setters.contains(name))
    throw std::runtime_error("Unknown setter name " + name);

  SetField::Setter setup;

  if (name == "SetUniformField") {
    LOG("  Using SetUniformField setter");
    Vector3R value = parse_vector(setter, "value");
    setup = SetUniformField(value);
    LOG("    Field value: {} {} {}", REP3_A(value));
  }
  else if (name == "SetGradientField") {
    LOG("  Using SetGradientField setter");
    Vector3R value = parse_vector(setter, "value");
    PetscReal length = setter.at("length").get<PetscReal>();

    std::string axis_name = "X";
    if (setter.contains("axis"))
      setter.at("axis").get_to(axis_name);

    PetscInt axis;
    if (axis_name == "X" || axis_name == "x")      axis = 0;
    else if (axis_name == "Y" || axis_name == "y") axis = 1;
    else if (axis_name == "Z" || axis_name == "z") axis = 2;
    else throw std::runtime_error("Unknown axis for SetGradientField: " + axis_name);

    PetscReal origin = 0.0;
    if (setter.contains("origin"))
      origin = setter.at("origin").get<PetscReal>();

    setup = SetGradientField(value, axis, length, origin);
    LOG("    Field value: {} {} {}, axis: {}, length: {}, origin: {}", REP3_A(value), axis_name, length, origin);
  }
  else if (name == "SetAzimuthalField") {
    LOG("  Using SetAzimuthalField setter");
    PetscReal value = setter.at("value").get<PetscReal>();
    Vector3R center = parse_vector(setter, "center");

    PetscReal radius = 0.0;
    if (setter.contains("radius"))
      radius = setter.at("radius").get<PetscReal>();

    setup = SetAzimuthalField(value, center[X], center[Y], radius);
    LOG("    Field value: {}, center: {} {}, radius: {}", value, center[X], center[Y], radius);
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
  else if (name == "SetCosineField") {
    LOG("  Using SetCosineField setter");
    Vector3R amplitude = parse_vector(setter, "amplitude");
    Vector3R wave_number = parse_vector(setter, "wave_number");
    Vector3R phase = setter.contains("phase")
      ? parse_vector(setter, "phase") : Vector3R{};

    LOG("    Cosine amplitude: {} {} {}", REP3_A(amplitude));
    LOG("    Cosine wave_number: {} {} {}", REP3_A(wave_number));
    LOG("    Cosine phase: {} {} {}", REP3_A(phase));

    setup = SetCosineField{amplitude, wave_number, phase};
  }
  else if (name == "SetGeneralCosineField") {
    LOG("  Using SetGeneralCosineField setter");
    Vector3R amplitude = parse_vector(setter, "amplitude");
    Vector3R wave_number = parse_vector(setter, "wave_number");

    LOG("    Cosine amplitude: {} {} {}", REP3_A(amplitude));
    LOG("    Cosine wave_number: {} {} {}", REP3_A(wave_number));

    setup = SetGeneralCosineField{amplitude, wave_number};
  }

  commands_.emplace_back(
    std::make_unique<SetField>(F, F0, std::move(setup)));

  LOG("  SetField command is added for {}", field);
  PetscFunctionReturn(PETSC_SUCCESS);
}
