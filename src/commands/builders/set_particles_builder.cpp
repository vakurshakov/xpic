#include "set_particles_builder.h"

SetParticlesBuilder::SetParticlesBuilder(
  interfaces::Simulation& simulation, std::vector<Command_up>& result)
  : ParticlesBuilder(simulation, result)
{
}

PetscErrorCode SetParticlesBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  std::vector<SortInfo> sort_infos{{
    info.at("particles").get<std::string>(),
    info.at("momentum"),
  }};

  // Paired loader: one shared coordinate stream for both sorts.
  if (info.contains("paired_with")) {
    sort_infos.push_back({
      info.at("paired_with").get<std::string>(),
      info.at("momentum_paired"),
    });
  }

  CoordinateGenerator generate_coordinate;
  PetscInt number_of_particles = 0;

  auto sorts = load_sorts(info.at("coordinate"), sort_infos,  //
    generate_coordinate, number_of_particles);

  commands_.emplace_back(std::make_unique<SetParticles>(
    std::move(sorts), number_of_particles, generate_coordinate));

  for (const auto& sort_info : sort_infos)
    LOG("  SetParticles command is added for \"{}\"", sort_info.name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

std::vector<SetParticles::Sort> SetParticlesBuilder::load_sorts(  //
  const Configuration::json_t& coordinate_info,                   //
  const std::vector<SortInfo>& sort_infos,                        //
  CoordinateGenerator& generate_coordinate,                       //
  PetscInt& number_of_particles)
{
  std::vector<SetParticles::Sort> sorts;
  for (const auto& sort_info : sort_infos) {
    auto&& particles = simulation_.get_named_particles(sort_info.name);

    // The first sort provides the number of particles
    if (sorts.empty()) {
      load_coordinate(coordinate_info, particles,  //
        generate_coordinate, number_of_particles);
    }

    MomentumGenerator generate_momentum;
    load_momentum(sort_info.momentum_info, particles, generate_momentum);
    sorts.push_back({particles, std::move(generate_momentum)});
  }
  return sorts;
}
