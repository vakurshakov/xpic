#include "inject_particles_builder.h"

#include "src/commands/inject_particles.h"

InjectParticlesBuilder::InjectParticlesBuilder(
  const interfaces::Simulation& simulation, std::list<Command_up>& result)
  : SetParticlesBuilder(simulation, result)
{
}

PetscErrorCode InjectParticlesBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  auto&& ionized_name = info.at("ionized").get<std::string>();
  auto&& ejected_name = info.at("ejected").get<std::string>();
  auto&& ionized = simulation_.get_named_particles(ionized_name);
  auto&& ejected = simulation_.get_named_particles(ejected_name);

  PetscInt injection_start = 0;
  PetscInt injection_end = 1;

  if (info.contains("injection_start"))
    injection_start = TO_STEP(info.at("injection_start").get<PetscReal>(), dt);

  if (info.contains("injection_end")) {
    const Configuration::json_t& value = info.at("injection_end");

    switch (value.type()) {
      case nlohmann::json::value_t::string:
        if (value.get<std::string>() == "geom_t")
          injection_end = geom_nt;
        break;
      default:
        injection_end = TO_STEP(info.at("injection_end").get<PetscReal>(), dt);
        break;
    }
  }

  PetscInt per_step_particles_num = 0.0;

  CoordinateGenerator generate_coordinate;
  load_coordinate(info.at("coordinate"), //
    ionized, generate_coordinate, per_step_particles_num);

  per_step_particles_num /= (injection_end - injection_start);

  if (info.contains("per_step_particles_num"))
    info.at("per_step_particles_num").get_to(per_step_particles_num);

  MomentumGenerator generate_momentum_i;
  momentum(info.at("momentum_i"), ionized, generate_momentum_i);

  MomentumGenerator generate_momentum_e;
  momentum(info.at("momentum_e"), ejected, generate_momentum_e);

  auto&& diag = std::make_unique<InjectParticles>(ionized, ejected,
    injection_start, injection_end, per_step_particles_num,  //
    generate_coordinate, generate_momentum_i, generate_momentum_e);

  commands_.emplace_back(std::move(diag));

  LOG("Inject particles command is added for ionized: \"{}\", ejected: \"{}\"", ionized_name, ejected_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}
