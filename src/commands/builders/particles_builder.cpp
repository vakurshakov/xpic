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
    number_of_particles = std::numbers::pi * POW2(cyl.radius) * cyl.height * frac;
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

void ParticlesBuilder::load_geometry(
  const Configuration::json_t& info, BoxGeometry& box)
{
  Vector3R min{0.0};
  Vector3R max{Geom};

  if (info.contains("min"))
    min = parse_vector(info, "min");
  if (info.contains("max"))
    max = parse_vector(info, "max");

  box = BoxGeometry(min, max);
}

void ParticlesBuilder::load_geometry(
  const Configuration::json_t& info, CylinderGeometry& cyl)
{
  Vector3R center{0.5 * geom_x, 0.5 * geom_y, 0.5 * geom_z};

  PetscReal s = std::min(geom_x, geom_y);
  PetscReal radius = 0.5 * std::hypot(s, s);
  PetscReal height = geom_z;

  if (info.contains("center"))
    center = parse_vector(info, "center");
  if (info.contains("radius"))
    info.at("radius").get_to(radius);
  if (info.contains("height"))
    info.at("height").get_to(height);

  cyl = CylinderGeometry(center, radius, height);
}

}  // namespace interfaces
