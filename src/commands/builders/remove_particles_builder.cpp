#include "remove_particles_builder.h"

#include "src/commands/remove_particles.h"

RemoveParticlesBuilder::RemoveParticlesBuilder(
  const interfaces::Simulation& simulation, std::list<Command_up>& result)
  : interfaces::ParticlesBuilder(simulation, result)
{
}

PetscErrorCode RemoveParticlesBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  auto&& particles_name = info.at("particles").get<std::string>();
  auto&& particles = simulation_.get_named_particles(particles_name);

  std::string name;
  RemoveParticles::Tester test;

  const Configuration::json_t& geometry = info.at("geometry");
  geometry.at("name").get_to(name);

  if (name == "BoxGeometry") {
    Vector3R min{0.0};
    Vector3R max{Geom};

    if (geometry.contains("min"))
      min = parse_vector(geometry, "min");
    if (geometry.contains("max"))
      max = parse_vector(geometry, "max");

    test = RemoveFromBox(BoxGeometry(min, max));
  }
  else if (name == "CylinderGeometry") {
    Vector3R center{0.5 * geom_x, 0.5 * geom_y, 0.5 * geom_z};

    PetscReal s = std::min(geom_x, geom_y);
    PetscReal radius = 0.5 * std::hypot(s, s);
    PetscReal height = geom_z;

    if (geometry.contains("center"))
      center = parse_vector(geometry, "center");
    if (geometry.contains("radius"))
      geometry.at("radius").get_to(radius);
    if (geometry.contains("height"))
      geometry.at("height").get_to(height);

    test = RemoveFromCylinder(CylinderGeometry(center, radius, height));
  }
  else {
    throw std::runtime_error("Unknown coordinate geometry name " + name);
  }

  commands_.emplace_back(
    std::make_unique<RemoveParticles>(particles, std::move(test)));

  LOG("RemoveParticles command is added for \"{}\"", particles_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}
