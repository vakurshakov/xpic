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
  std::unique_ptr<RemoveParticles> diag;

  const Configuration::json_t& geometry = info.at("geometry");
  geometry.at("name").get_to(name);

  if (name == "BoxGeometry") {
    Vector3R min = parse_vector(geometry, "min");
    Vector3R max = parse_vector(geometry, "max");
    diag = std::make_unique<RemoveParticles>(particles, BoxGeometry(min, max));
  }
  else {
    throw std::runtime_error("Unknown coordinate geometry name " + name);
  }

  commands_.emplace_back(std::move(diag));

  LOG("RemoveParticles command is added for \"{}\"", particles_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}
