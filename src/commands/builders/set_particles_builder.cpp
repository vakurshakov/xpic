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
  load_coordinate(info.at("coordinate"),  //
    particles, generate_coordinate, number_of_particles);

  auto&& coordinate_name = info.at("coordinate").at("name").get<std::string>();
  bool paired_coordinate = (coordinate_name == "KineticIonSoundCoordinates");

  auto validate_paired_loaders = [&](const Configuration::json_t& momentum_info)
  {
    const auto momentum_name = momentum_info.at("name").get<std::string>();
    const bool paired_momentum = momentum_name == "KineticIonSoundMoments";
    if (paired_coordinate != paired_momentum)
      throw std::runtime_error(
        "paired coordinate and momentum generators must be used together");

    if (momentum_name != "KineticIonSoundMoments")
      return;
    if (coordinate_name != "KineticIonSoundCoordinates")
      throw std::runtime_error(
        "KineticIonSoundMoments requires KineticIonSoundCoordinates");

    const auto& coordinate_info = info.at("coordinate");
    const auto vector_or_zero =
      [&](const Configuration::json_t& block, const char* key)
    {
      return block.contains(key) ? parse_vector(block, key) : Vector3R{};
    };
    const auto require_same =
      [](const Vector3R& left, const Vector3R& right, const char* what)
    {
      if ((left - right).abs_max() != 0.0)
        throw std::runtime_error(std::format(
          "KineticIonSoundMoments {} must match its coordinate loader", what));
    };
    require_same(parse_vector(momentum_info, "min"),
      parse_vector(coordinate_info, "min"), "box minimum");
    require_same(parse_vector(momentum_info, "max"),
      parse_vector(coordinate_info, "max"), "box maximum");
    require_same(parse_vector(momentum_info, "wave_number"),
      parse_vector(coordinate_info, "wave_number"), "wave number");
    require_same(parse_vector(momentum_info, "density_amplitude"),
      parse_vector(coordinate_info, "amplitude"), "density amplitude");
    require_same(vector_or_zero(momentum_info, "density_phase"),
      vector_or_zero(coordinate_info, "phase"), "density phase");
  };

  // Paired loader: one shared coordinate stream for both sorts. Triggered by
  // `"paired_with": "<other_sort_name>"` plus a second momentum generator
  // `"momentum_paired"`.
  if (info.contains("paired_with")) {
    auto&& other_name = info.at("paired_with").get<std::string>();
    auto&& other = simulation_.get_named_particles(other_name);

    validate_paired_loaders(info.at("momentum"));
    validate_paired_loaders(info.at("momentum_paired"));

    MomentumGenerator generate_momentum_a;
    load_momentum(info.at("momentum"), particles, generate_momentum_a);

    MomentumGenerator generate_momentum_b;
    load_momentum(info.at("momentum_paired"), other, generate_momentum_b);

    commands_.emplace_back(std::make_unique<SetParticles>(
      std::vector<SetParticles::Sort>{
        {particles, generate_momentum_a},
        {other, generate_momentum_b},
      },
      number_of_particles, generate_coordinate));

    LOG("  SetParticles command is added for \"{}\" + \"{}\"", particles_name, other_name);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  validate_paired_loaders(info.at("momentum"));
  MomentumGenerator generate_momentum;
  load_momentum(info.at("momentum"), particles, generate_momentum);

  commands_.emplace_back(std::make_unique<SetParticles>(
    std::vector<SetParticles::Sort>{
      {particles, generate_momentum},
    },
    number_of_particles,  //
    generate_coordinate   //
    ));

  LOG("  SetParticles command is added for \"{}\"", particles_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}
