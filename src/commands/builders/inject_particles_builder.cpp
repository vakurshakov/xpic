#include "inject_particles_builder.h"

#include "src/commands/inject_particles.h"

InjectParticlesBuilder::InjectParticlesBuilder(
  const interfaces::Simulation& simulation, std::list<Command_up>& result)
  : CommandBuilder(simulation, result)
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
  load_coordinate(info.at("set_point_of_birth"), ionized, generate_coordinate, per_step_particles_num);

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


/// @todo This should be reused across particles setups
void InjectParticlesBuilder::load_coordinate(const Configuration::json_t& info,
  const interfaces::Particles& particles, CoordinateGenerator& gen, PetscInt& per_step_particles_num)
{
  const PetscInt Npi = particles.parameters().Np;

  if (info.at("name") == "CoordinateInBox") {
    Vector3R min{0.0};
    Vector3R max{Geom};
    gen = CoordinateInBox(min, max);
    per_step_particles_num =
      (max - min).elements_product() / (dx * dy * dz) * Npi;
  }
}


void InjectParticlesBuilder::load_momentum(const Configuration::json_t& info,
  const interfaces::Particles& particles, MomentumGenerator& gen)
{
  if (info.at("name") == "MaxwellianMomentum") {
    bool tov = false;

    if (info.contains("tov"))
      info.at("tov").get_to(tov);

    gen = MaxwellianMomentum(particles.parameters(), tov);
  }
}
