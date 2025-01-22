#include "fields_damping_builder.h"

#include "src/commands/fields_damping.h"

FieldsDampingBuilder::FieldsDampingBuilder(
  const interfaces::Simulation& simulation, std::list<Command_up>& result)
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

  std::unique_ptr<FieldsDamping> diag;
  const Configuration::json_t& geometry = info.at("geometry");

  std::string name;
  geometry.at("name").get_to(name);

  if (name == "BoxGeometry") {
    Vector3R min = parse_vector(geometry, "min");
    Vector3R max = parse_vector(geometry, "max");
    diag = std::make_unique<FieldsDamping>(simulation_.world_.da, E, B, B0,
      BoxGeometry(min, max), damping_coefficient);
  }
  else {
    throw std::runtime_error("Unknown coordinate generator name " + name);
  }

  commands_.emplace_back(std::move(diag));

  LOG("FieldsDamping command is added");
  PetscFunctionReturn(PETSC_SUCCESS);
}
