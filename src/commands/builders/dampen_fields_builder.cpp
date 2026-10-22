#include "dampen_fields_builder.h"

#include "src/commands/dampen_fields.h"

DampenFieldsBuilder::DampenFieldsBuilder(
  interfaces::Simulation& simulation, std::vector<Command_up>& result)
  : CommandBuilder(simulation, result)
{
}

PetscErrorCode DampenFieldsBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  Vec E = simulation_.E;
  Vec B = simulation_.B;
  Vec B0 = simulation_.B0;

  PetscReal damping_coefficient;
  info.at("damping_coefficient").get_to(damping_coefficient);

  const Configuration::json_t& geometry = info.at("geometry");

  std::string name;
  geometry.at("name").get_to(name);

  DampenFields::Tester test;
  DampenFields::Damping damp;

  if (name == "BoxGeometry") {
    BoxGeometry box;
    load_geometry(geometry, box);
    test = WithinBox(box);
    damp = DampForBox(std::move(box), damping_coefficient);
  }
  else if (name == "CylinderGeometry") {
    CylinderGeometry cyl;
    load_geometry(geometry, cyl);
    test = WithinCylinder(cyl);
    damp = DampForCylinder(std::move(cyl), damping_coefficient);
  }
  else {
    throw std::runtime_error("Unknown geometry name " + name);
  }

  commands_.emplace_back(std::make_unique<DampenFields>(
    simulation_.world.da, E, B, B0, std::move(test), std::move(damp)));

  LOG("  DampenFields command is added");
  PetscFunctionReturn(PETSC_SUCCESS);
}
