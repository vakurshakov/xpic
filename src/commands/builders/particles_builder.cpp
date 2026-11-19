#include "particles_builder.h"

#include "src/utils/vector3.h"

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
  const PetscReal np = Np / (dx * dy * dz);

  if (name == "PreciseCoordinate") {
    auto dots = parse_vectors(info, "value");
    number_of_particles = (PetscInt)dots.size();
    gen = PreciseCoordinate(std::move(dots));
  }
  else if (name == "CoordinateInBox" || name == "CoordinateInBoxQS") {
    BoxGeometry box;
    load_geometry(info, box);
    number_of_particles = std::round((box.max - box.min).elements_product() * np);
    if (name == "CoordinateInBoxQS")
      gen = CoordinateInBoxQS{box};
    else
      gen = CoordinateInBox{box};
  }
  else if (name == "CoordinateInCylinder" || name == "CoordinateInCylinderQS") {
    CylinderGeometry cyl;
    load_geometry(info, cyl);
    number_of_particles = std::round(M_PI * POW2(cyl.radius) * cyl.height * np);
    if (name == "CoordinateInCylinderQS")
      gen = CoordinateInCylinderQS{cyl};
    else
      gen = CoordinateInCylinder{cyl};
  }
  else if (name == "CoordinateInBoxCosineHump" ||
    name == "CoordinateInBoxCosineHumpQS") {
    BoxGeometry box;
    load_geometry(info, box);

    std::string axis_name = info.value("axis", "Z");
    Axis axis = get_component(axis_name);

    number_of_particles = std::round((box.max - box.min).elements_product() * np);
    if (name == "CoordinateInBoxCosineHumpQS")
      gen = CoordinateInBoxCosineHumpQS{box, axis};
    else
      gen = CoordinateInBoxCosineHump{box, axis};
  }
  else {
    throw std::runtime_error("Unknown coordinate generator name " + name);
  }
}

void ParticlesBuilder::load_momentum(const Configuration::json_t& info,
  const interfaces::Particles& particles, PetscInt number_of_particles,
  MomentumGenerator& gen)
{
  std::string name;
  info.at("name").get_to(name);

  if (name == "PreciseMomentum") {
    auto values = parse_vectors(info, "value");

    if (!(values.size() == 1 || (PetscInt)values.size() == number_of_particles)) {
      throw std::runtime_error(
        std::format("PreciseMomentum has {} values for {} particles",
          values.size(), number_of_particles));
    }

    gen = PreciseMomentum(std::move(values));
  }
  else if (name == "MaxwellianMomentum") {
    bool tov = false;

    if (info.contains("tov"))
      info.at("tov").get_to(tov);

    gen = MaxwellianMomentum(particles.parameters, tov);
  }
  else if (name == "MaxwellShiftedCosine") {
    BoxGeometry box;
    load_geometry(info, box);

    Vector3R velocity = parse_vector(info, "velocity");
    Vector3R wave_number = parse_vector(info, "wave_number");

    Vector3R phase;
    if (info.contains("phase"))
      phase = parse_vector(info, "phase");

    gen = MaxwellShiftedCosine(
      particles.parameters, box, velocity, wave_number, phase);
  }
  else if (name == "MaxwellShiftedSine") {
    BoxGeometry box;
    load_geometry(info, box);

    Vector3R velocity = parse_vector(info, "velocity");
    Vector3R wave_number = parse_vector(info, "wave_number");

    Vector3R phase;
    if (info.contains("phase"))
      phase = parse_vector(info, "phase");

    gen = MaxwellShiftedSine(
      particles.parameters, box, velocity, wave_number, phase);
  }
  else {
    throw std::runtime_error("Unknown momentum generator name " + name);
  }
}
