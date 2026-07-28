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
  else if (name == "CoordinateInBoxSineDensity") {
    BoxGeometry box;
    load_geometry(info, box);

    Vector3R amplitude = parse_vector(info, "amplitude");
    Vector3R wave_number = parse_vector(info, "wave_number");

    number_of_particles = (box.max - box.min).elements_product() * frac;
    gen = CoordinateInBoxSineDensity(std::move(box), amplitude, wave_number);
  }
  else if (name == "CoordinateInBoxDisplacedSine") {
    BoxGeometry box;
    load_geometry(info, box);

    Vector3R amplitude = parse_vector(info, "amplitude");
    Vector3R wave_number = parse_vector(info, "wave_number");

    number_of_particles = (box.max - box.min).elements_product() * frac;
    gen = CoordinateInBoxDisplacedSine(std::move(box), amplitude, wave_number);
  }
  else if (name == "CoordinateInBoxQuietSine") {
    BoxGeometry box;
    load_geometry(info, box);

    Vector3R amplitude = parse_vector(info, "amplitude");
    Vector3R wave_number = parse_vector(info, "wave_number");
    Vector3R phase = info.contains("phase") ? parse_vector(info, "phase") : Vector3R{0.0, 0.0, 0.0};

    number_of_particles = (box.max - box.min).elements_product() * frac;
    gen = CoordinateInBoxQuietSine{std::move(box), amplitude, wave_number, phase, 0};
  }
  else if (name == "CoordinateInBoxQuietSinePaired") {
    BoxGeometry box;
    load_geometry(info, box);

    Vector3R amplitude = parse_vector(info, "amplitude");
    Vector3R wave_number = parse_vector(info, "wave_number");
    Vector3R phase = info.contains("phase") ? parse_vector(info, "phase") : Vector3R{0.0, 0.0, 0.0};

    number_of_particles = (box.max - box.min).elements_product() * frac;
    if (number_of_particles % 2 != 0)
      throw std::runtime_error(
        "CoordinateInBoxQuietSinePaired requires an even number of particles");
    gen = CoordinateInBoxQuietSinePaired{
      std::move(box), amplitude, wave_number, phase};
  }
  else if (name == "CoordinateInBoxQuiet") {
    BoxGeometry box;
    load_geometry(info, box);

    number_of_particles = (box.max - box.min).elements_product() * frac;
    gen = CoordinateInBoxQuiet{std::move(box), Np, 0};
  }
  else if (name == "CoordinateInCylinderCosineHump") {
    CylinderGeometry cyl;
    load_geometry(info, cyl);
    number_of_particles = M_PI * POW2(cyl.radius) * cyl.height * frac;
    gen = CoordinateInCylinderCosineHump(std::move(cyl));
  }
  else if (name == "CoordinateInBoxCosineHump") {
    BoxGeometry box;
    load_geometry(info, box);

    std::string axis_name = "Z";
    if (info.contains("axis"))
      info.at("axis").get_to(axis_name);

    Axis axis;
    if (axis_name == "X" || axis_name == "x")      axis = X;
    else if (axis_name == "Y" || axis_name == "y") axis = Y;
    else if (axis_name == "Z" || axis_name == "z") axis = Z;
    else throw std::runtime_error("Unknown axis for CoordinateInBoxCosineHump: " + axis_name);

    number_of_particles = (box.max - box.min).elements_product() * frac;
    gen = CoordinateInBoxCosineHump{std::move(box), axis, 0};
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
  else if (name == "MaxwellCosinePerturbation") {
    BoxGeometry box;
    load_geometry(info, box);

    Vector3R a = parse_vector(info, "amplitude");
    Vector3R m = parse_vector(info, "wave_number");

    gen = MaxwellCosinePerturbation(particles.parameters, box, a, m);
  }
  else if (name == "MaxwellShiftedSine") {
    BoxGeometry box;
    load_geometry(info, box);

    Vector3R velocity = parse_vector(info, "velocity");
    Vector3R wave_number = parse_vector(info, "wave_number");
    Vector3R phase = info.contains("phase") ? parse_vector(info, "phase") : Vector3R{0.0, 0.0, 0.0};

    gen = MaxwellShiftedSine(particles.parameters, box, velocity, wave_number, phase);
  }
  else if (name == "MaxwellShiftedSineQuiet") {
    BoxGeometry box;
    load_geometry(info, box);

    Vector3R velocity = parse_vector(info, "velocity");
    Vector3R wave_number = parse_vector(info, "wave_number");
    Vector3R phase = info.contains("phase") ? parse_vector(info, "phase") : Vector3R{0.0, 0.0, 0.0};

    gen = MaxwellShiftedSineQuiet(
      particles.parameters, box, velocity, wave_number, phase);
  }
  else {
    throw std::runtime_error("Unknown momentum generator name " + name);
  }
}
