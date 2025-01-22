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
  load_coordinate(
    info.at("coordinate"), particles, generate_coordinate, number_of_particles);

  MomentumGenerator generate_momentum;
  momentum(info.at("momentum"), particles, generate_momentum);

  auto&& diag = std::make_unique<SetParticles>(
    particles, number_of_particles, generate_coordinate, generate_momentum);

  commands_.emplace_back(std::move(diag));

  LOG("Set particles command is added for particles: \"{}\"", particles_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}


void SetParticlesBuilder::load_coordinate(const Configuration::json_t& info,
  const interfaces::Particles& particles, CoordinateGenerator& gen,
  PetscInt& number_of_particles)
{
  std::string name;
  info.at("name").get_to(name);

  const PetscInt Npi = particles.parameters().Np;

  if (name == "PreciseCoordinate") {
    gen = PreciseCoordinate(parse_vector(info, "value"));
    number_of_particles = Npi;
  }
  else if (name == "CoordinateInBox") {
    Vector3R min{0.0};
    Vector3R max{Geom};
    gen = CoordinateInBox(min, max);
    number_of_particles = (max - min).elements_product() / (dx * dy * dz) * Npi;
  }
  else {
    throw std::runtime_error("Unknown coordinate generator name " + name);
  }
}

void SetParticlesBuilder::momentum(const Configuration::json_t& info,
  const interfaces::Particles& particles, MomentumGenerator& gen)
{
  std::string name;
  info.at("name").get_to(name);

  if (name == "PreciseMomentum") {
    gen = PreciseMomentum(parse_vector(info, "value"));
  }
  else if (name == "MaxwellianMomentum") {
    bool tov = false;

    if (info.contains("tov"))
      info.at("tov").get_to(tov);

    gen = MaxwellianMomentum(particles.parameters(), tov);
  }
  else {
    throw std::runtime_error("Unknown coordinate generator name " + name);
  }
}
