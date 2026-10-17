#include "inject_particles_builder.h"

#include "src/commands/inject_particles.h"

InjectParticlesBuilder::InjectParticlesBuilder(
  interfaces::Simulation& simulation, std::vector<Command_up>& result)
  : SetParticlesBuilder(simulation, result)
{
}

PetscErrorCode InjectParticlesBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  auto&& ionized_name = info.at("ionized").get<std::string>();
  auto&& ejected_name = info.at("ejected").get<std::string>();

  PetscInt number_of_particles = 0;
  CoordinateGenerator generate_coordinate;

  auto sorts = load_sorts(info.at("coordinate"),
    {
      {ionized_name, info.at("momentum_i")},
      {ejected_name, info.at("momentum_e")},
    },
    generate_coordinate, number_of_particles);

  /// @note Since we can use this as a quasi-neutral
  /// start, we choose t in [0, 1] by default
  PetscInt injection_start = 0;
  PetscInt injection_end = 1;

  if (info.contains("injection_start"))
    injection_start = ROUND_STEP(parse_value(info.at("injection_start")), dt);

  if (info.contains("injection_end"))
    injection_end = ROUND_STEP(parse_value(info.at("injection_end")), dt);

  PetscInt tau = std::max(injection_end - injection_start, 1);

  if (info.contains("tau"))
    tau = ROUND_STEP(parse_value(info.at("tau")), dt);

  PetscInt per_step_particles_num = number_of_particles / tau;

  if (info.contains("per_step_particles_num"))
    info.at("per_step_particles_num").get_to(per_step_particles_num);

  commands_.emplace_back(std::make_unique<InjectParticles>(std::move(sorts),
    injection_start, injection_end, per_step_particles_num, generate_coordinate));

  LOG("  InjectParticles command is added with ionized: \"{}\", ejected: \"{}\"", ionized_name, ejected_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}
