#include "set_particles_builder.h"

#include "src/commands/set_particles.h"
#include "src/impls/drift_kinetic/particles.h"
#include "src/impls/drift_kinetic/set_paired_particles.h"

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

  const auto coordinate_name =
    info.at("coordinate").at("name").get<std::string>();
  const bool quiet_coordinate =
    coordinate_name == "CoordinateInBoxQuietSinePaired" ||
    coordinate_name == "CoordinateInBoxQuietSineExactPaired" ||
    coordinate_name == "CoordinateInBoxQuietSineExactLatticePaired";
  auto validate_quiet_pair = [this, quiet_coordinate, coordinate_name, &info](
                               const Configuration::json_t& momentum_info) {
    const auto momentum_name = momentum_info.at("name").get<std::string>();
    const bool quiet_momentum =
      momentum_name == "MaxwellShiftedSineQuiet" ||
      momentum_name == "MaxwellianVelocityQuiet" ||
      momentum_name == "KineticIonSoundMomentsQuiet" ||
      momentum_name == "KineticIonSoundQuiet";
    if (quiet_coordinate != quiet_momentum)
      throw std::runtime_error(
        "paired quiet coordinate and momentum generators must be used together");

    if (momentum_name != "KineticIonSoundMomentsQuiet")
      return;
    if (coordinate_name != "CoordinateInBoxQuietSineExactLatticePaired")
      throw std::runtime_error(
        "KineticIonSoundMomentsQuiet requires CoordinateInBoxQuietSineExactLatticePaired");

    const auto& coordinate_info = info.at("coordinate");
    const auto vector_or_zero = [this](const Configuration::json_t& block,
                                  const char* key) {
      return block.contains(key) ? parse_vector(block, key) : Vector3R{};
    };
    const auto require_same = [](const Vector3R& left,
                              const Vector3R& right, const char* what) {
      if ((left - right).abs_max() != 0.0)
        throw std::runtime_error(std::format(
          "KineticIonSoundMomentsQuiet {} must match its coordinate loader",
          what));
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

  // Drift-kinetic paired loader: one shared coordinate stream for both sorts.
  // Triggered by `"paired_with": "<other_sort_name>"` plus a second momentum
  // generator `"momentum_paired"`. Both sorts must be `drift_kinetic::Particles`.
  // The command is emitted only once — when the second sort's `SetParticles`
  // entry is encountered, it is silently skipped (an empty no-op is added so
  // the dispatch order stays predictable).
  if (info.contains("paired_with")) {
    auto&& other_name = info.at("paired_with").get<std::string>();
    auto&& other = simulation_.get_named_particles(other_name);

    auto* dk_a = dynamic_cast<drift_kinetic::Particles*>(&particles);
    auto* dk_b = dynamic_cast<drift_kinetic::Particles*>(&other);

    if (dk_a && dk_b) {
      validate_quiet_pair(info.at("momentum"));
      validate_quiet_pair(info.at("momentum_paired"));

      MomentumGenerator generate_momentum_a;
      load_momentum(info.at("momentum"), particles, generate_momentum_a);

      MomentumGenerator generate_momentum_b;
      load_momentum(info.at("momentum_paired"), other, generate_momentum_b);

      auto&& diag = std::make_unique<drift_kinetic::SetPairedParticles>(
        *dk_a, *dk_b, number_of_particles, generate_coordinate,
        generate_momentum_a, generate_momentum_b);

      commands_.emplace_back(std::move(diag));

      LOG("  SetPairedParticles command is added for \"{}\" + \"{}\"",
        particles_name, other_name);
      PetscFunctionReturn(PETSC_SUCCESS);
    }

    throw std::runtime_error(
      "SetParticles \"paired_with\" requires both sorts to be drift_kinetic");
  }

  validate_quiet_pair(info.at("momentum"));
  MomentumGenerator generate_momentum;
  load_momentum(info.at("momentum"), particles, generate_momentum);

  auto&& diag = std::make_unique<SetParticles>(
    particles, number_of_particles, generate_coordinate, generate_momentum);

  commands_.emplace_back(std::move(diag));

  LOG("  SetParticles command is added for \"{}\"", particles_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}
