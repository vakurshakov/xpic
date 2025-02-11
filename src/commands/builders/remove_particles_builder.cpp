#include "remove_particles_builder.h"

#include "src/commands/remove_particles.h"

RemoveParticlesBuilder::RemoveParticlesBuilder(
  const interfaces::Simulation& simulation, std::vector<Command_up>& result)
  : interfaces::ParticlesBuilder(simulation, result)
{
}

PetscErrorCode RemoveParticlesBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  auto&& particles_name = info.at("particles").get<std::string>();
  auto&& particles = simulation_.get_named_particles(particles_name);

  const Configuration::json_t& geometry = info.at("geometry");

  std::string name;
  geometry.at("name").get_to(name);

  RemoveParticles::Tester test;

  if (name == "BoxGeometry") {
    BoxGeometry box;
    load_geometry(geometry, box);
    test = WithinBox(std::move(box));
  }
  else if (name == "CylinderGeometry") {
    CylinderGeometry cyl;
    load_geometry(geometry, cyl);
    test = WithinCylinder(std::move(cyl));
  }
  else {
    throw std::runtime_error("Unknown geometry name " + name);
  }

  commands_.emplace_back(
    std::make_unique<RemoveParticles>(particles, std::move(test)));

  LOG("  RemoveParticles command is added for \"{}\"", particles_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}
