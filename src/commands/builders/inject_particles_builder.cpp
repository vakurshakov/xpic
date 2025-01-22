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

  PetscInt per_step_particles_num = 0.0;

  CoordinateGenerator generate_coordinate;
  load_coordinate(info.at("set_point_of_birth"), //
    ionized, generate_coordinate, per_step_particles_num);

  MomentumGenerator generate_vi;
  load_momentum(info.at("load_momentum_i"), ionized, generate_vi);

  MomentumGenerator generate_ve;
  load_momentum(info.at("load_momentum_e"), ejected, generate_ve);

  auto&& diag = std::make_unique<InjectParticles>(  //
    ionized, ejected, 0, 1, per_step_particles_num, //
    generate_coordinate, generate_vi, generate_ve);

  commands_.emplace_back(std::move(diag));

  LOG("Inject particles command is added for ionized: \"{}\", ejected: \"{}\"", ionized_name, ejected_name);
  PetscFunctionReturn(PETSC_SUCCESS);
}
