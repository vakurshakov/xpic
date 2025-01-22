#include "set_particles_builder.h"

#include "src/commands/set_particles.h"

SetParticlesBuilder::SetParticlesBuilder(
  const interfaces::Simulation& simulation, std::list<Command_up>& result)
  : CommandBuilder(simulation, result)
{
}

PetscErrorCode SetParticlesBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  auto&& particles_name = info.at("particles").get<std::string>();
  auto&& particles = simulation_.get_named_particles(particles_name);

  PetscInt number_of_particles = 0.0;

  CoordinateGenerator generate_coordinate;
  load_coordinate(info.at("set_point_of_birth"), //
    particles, generate_coordinate, number_of_particles);

  MomentumGenerator generate_v;
  load_momentum(info.at("load_momentum"), particles, generate_v);

  auto&& diag = std::make_unique<SetParticles>(
    particles, number_of_particles, generate_coordinate, generate_v);

  commands_.emplace_back(std::move(diag));

  LOG("Set particles command is added for particles: \"{}\"", particles_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}


void SetParticlesBuilder::load_coordinate(const Configuration::json_t& info,
  const interfaces::Particles& particles, CoordinateGenerator& gen,
  PetscInt& number_of_particles)
{
  const PetscInt Npi = particles.parameters().Np;

  if (info.at("name") == "CoordinateInBox") {
    Vector3R min{0.0};
    Vector3R max{Geom};
    gen = CoordinateInBox(min, max);
    number_of_particles = (max - min).elements_product() / (dx * dy * dz) * Npi;
  }
}

void SetParticlesBuilder::load_momentum(const Configuration::json_t& info,
  const interfaces::Particles& particles, MomentumGenerator& gen)
{
  if (info.at("name") == "MaxwellianMomentum") {
    bool tov = false;

    if (info.contains("tov"))
      info.at("tov").get_to(tov);

    gen = MaxwellianMomentum(particles.parameters(), tov);
  }
}
