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

  MomentumGenerator generate_momentum;
  load_momentum(info.at("momentum"), particles, generate_momentum);

  auto&& diag = std::make_unique<SetParticles>(
    particles, number_of_particles, generate_coordinate, generate_momentum);

  commands_.emplace_back(std::move(diag));

  LOG("  SetParticles command is added for \"{}\"", particles_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}
