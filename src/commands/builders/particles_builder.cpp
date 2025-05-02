#include "particles_builder.h"

ParticlesBuilder::ParticlesBuilder(
  interfaces::Simulation& simulation, std::vector<Command_up>& result)
  : CommandBuilder(simulation, result)
{
}

void ParticlesBuilder::load_coordinate(const Configuration::json_t& info,
  const interfaces::Particles& particles, CoordinateGenerator& gen,
  PetscInt& number_of_particles)
{
  std::string name;
  info.at("name").get_to(name);

  const PetscInt Np = particles.parameters.Np;
  const PetscReal frac = Np / (dx * dy * dz);

  if (name == "PreciseCoordinate") {
    number_of_particles = Np;
    gen = PreciseCoordinate(parse_vector(info, "value"));
  }
  else if (name == "CoordinateInBox") {
    BoxGeometry box;
    load_geometry(info, box);
    number_of_particles = (box.max - box.min).elements_product() * frac;
    gen = CoordinateInBox(std::move(box));
  }
  else if (name == "CoordinateInCylinder") {
    CylinderGeometry cyl;
    load_geometry(info, cyl);
    number_of_particles = M_PI * POW2(cyl.radius) * cyl.height * frac;
    gen = CoordinateInCylinder(std::move(cyl));
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
