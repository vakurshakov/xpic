#include "fields_damping_builder.h"

#include "src/commands/fields_damping.h"

FieldsDampingBuilder::FieldsDampingBuilder(
  const interfaces::Simulation& simulation, std::vector<Command_up>& result)
  : CommandBuilder(simulation, result)
{
}

PetscErrorCode FieldsDampingBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  Vec E = simulation_.get_named_vector(info.at("E").get<std::string>());
  Vec B = simulation_.get_named_vector(info.at("B").get<std::string>());
  Vec B0 = simulation_.get_named_vector(info.at("B0").get<std::string>());

  PetscReal damping_coefficient;
  info.at("damping_coefficient").get_to(damping_coefficient);

  const Configuration::json_t& geometry = info.at("geometry");

  std::string name;
  geometry.at("name").get_to(name);

  FieldsDamping::Tester test;
  FieldsDamping::Damping damp;

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

  commands_.emplace_back(std::make_unique<FieldsDamping>(
    simulation_.world.da, E, B, B0, std::move(test), std::move(damp)));

  LOG("  FieldsDamping command is added");
  PetscFunctionReturn(PETSC_SUCCESS);
}
