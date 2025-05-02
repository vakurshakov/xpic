#include "set_particles_builder.h"

#include "src/commands/set_particles.h"

SetParticlesBuilder::SetParticlesBuilder(
  interfaces::Simulation& simulation, std::vector<Command_up>& result)
  : ParticlesBuilder(simulation, result)
{
}

PetscErrorCode SetParticlesBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  auto&& particles_name = info.at("particles").get<std::string>();
  auto&& particles = simulation_.get_named_particles(particles_name);

  PetscInt number_of_particles = 0.0;

  CoordinateGenerator generate_coordinate;
  load_coordinate(
    info.at("coordinate"), particles, generate_coordinate, number_of_particles);

  MomentumGenerator generate_momentum;
  load_momentum(info.at("momentum"), particles, generate_momentum);

  auto&& diag = std::make_unique<SetParticles>(
    particles, number_of_particles, generate_coordinate, generate_momentum);

  commands_.emplace_back(std::move(diag));

  LOG("  SetParticles command is added for \"{}\"", particles_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}
