#include "particles_builder.h"

namespace interfaces {

ParticlesBuilder::ParticlesBuilder(
  const interfaces::Simulation& simulation, std::list<Command_up>& result)
  : CommandBuilder(simulation, result)
{
}

void ParticlesBuilder::load_coordinate(const Configuration::json_t& info,
  const interfaces::Particles& particles, CoordinateGenerator& gen,
  PetscInt& number_of_particles)
{
  std::string name;
  info.at("name").get_to(name);

  const PetscInt Npi = particles.parameters.Np;

  if (name == "PreciseCoordinate") {
    gen = PreciseCoordinate(parse_vector(info, "value"));
    number_of_particles = Npi;
  }
  else if (name == "CoordinateInBox") {
    Vector3R min = parse_vector(info, "min");
    Vector3R max = parse_vector(info, "max");
    gen = CoordinateInBox(BoxGeometry(min, max));
    number_of_particles = (max - min).elements_product() / (dx * dy * dz) * Npi;
  }
  else {
    throw std::runtime_error("Unknown coordinate generator name " + name);
  }
}

void ParticlesBuilder::load_momentum(const Configuration::json_t& info,
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

    gen = MaxwellianMomentum(particles.parameters, tov);
  }
  else {
    throw std::runtime_error("Unknown coordinate generator name " + name);
  }
}

}  // namespace interfaces
