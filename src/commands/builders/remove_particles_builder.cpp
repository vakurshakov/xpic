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
    BoxGeometry box;
    load_geometry(geometry, box);
    test = RemoveFromBox(std::move(box));
  }
  else if (name == "CylinderGeometry") {
    CylinderGeometry cyl;
    load_geometry(geometry, cyl);
    test = RemoveFromCylinder(std::move(cyl));
  }
  else {
    throw std::runtime_error("Unknown coordinate geometry name " + name);
  }

  commands_.emplace_back(
    std::make_unique<RemoveParticles>(particles, std::move(test)));

  LOG("RemoveParticles command is added for \"{}\"", particles_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}
