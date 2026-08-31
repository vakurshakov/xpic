#include "inject_particles_builder.h"

#include "src/commands/inject_particles.h"
#include "src/impls/drift_kinetic/inject_particles.h"
#include "src/impls/drift_kinetic/particles.h"

InjectParticlesBuilder::InjectParticlesBuilder(
  interfaces::Simulation& simulation, std::vector<Command_up>& result)
  : ParticlesBuilder(simulation, result)
{
}

PetscErrorCode InjectParticlesBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  auto&& ionized_name = info.at("ionized").get<std::string>();
  auto&& ejected_name = info.at("ejected").get<std::string>();
  auto&& ionized = simulation_.get_named_particles(ionized_name);
  auto&& ejected = simulation_.get_named_particles(ejected_name);

  for (const char* key : {"momentum_i", "momentum_e"}) {
    if (info.at(key).at("name").get<std::string>() ==
        "KineticIonSoundMomentsQuiet")
      throw std::runtime_error(
        "KineticIonSoundMomentsQuiet is valid only for a paired SetParticles command");
  }

  /// @note Since we can use this as a quasi-neutral
  /// start, we choose t in [0, 1] by default
  PetscInt injection_start = 0;
  PetscInt injection_end = 1;

  if (info.contains("injection_start"))
    injection_start = ROUND_STEP(parse_value(info.at("injection_start")), dt);

  if (info.contains("injection_end"))
    injection_end = ROUND_STEP(parse_value(info.at("injection_end")), dt);

  PetscInt per_step_particles_num = 0.0;

  CoordinateGenerator generate_coordinate;
  load_coordinate(info.at("coordinate"), //
    ionized, generate_coordinate, per_step_particles_num);

  PetscInt tau = (injection_end - injection_start);

  if (info.contains("tau"))
    tau = ROUND_STEP(parse_value(info.at("tau")), dt);

  per_step_particles_num /= tau;

  if (info.contains("per_step_particles_num"))
    info.at("per_step_particles_num").get_to(per_step_particles_num);

  MomentumGenerator generate_momentum_i;
  load_momentum(info.at("momentum_i"), ionized, generate_momentum_i);

  MomentumGenerator generate_momentum_e;
  load_momentum(info.at("momentum_e"), ejected, generate_momentum_e);

  auto* dk_ionized = dynamic_cast<drift_kinetic::Particles*>(&ionized);
  auto* dk_ejected = dynamic_cast<drift_kinetic::Particles*>(&ejected);

  if (dk_ionized && dk_ejected) {
    auto&& diag = std::make_unique<drift_kinetic::InjectParticles>(
      *dk_ionized, *dk_ejected, injection_start, injection_end,
      per_step_particles_num, generate_coordinate, generate_momentum_i,
      generate_momentum_e);
    commands_.emplace_back(std::move(diag));
  }
  else {
    auto&& diag = std::make_unique<InjectParticles>(ionized, ejected,
      injection_start, injection_end, per_step_particles_num,  //
      generate_coordinate, generate_momentum_i, generate_momentum_e);
    commands_.emplace_back(std::move(diag));
  }

  LOG("  InjectParticles command is added with ionized: \"{}\", ejected: \"{}\"", ionized_name, ejected_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}
