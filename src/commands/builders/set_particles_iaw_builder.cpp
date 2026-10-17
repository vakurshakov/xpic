#include "set_particles_iaw_builder.h"

#include "src/commands/set_particles_iaw.h"

SetParticlesIonAcousticWaveBuilder::SetParticlesIonAcousticWaveBuilder(
  interfaces::Simulation& simulation, std::vector<Command_up>& result)
  : CommandBuilder(simulation, result)
{
}

PetscErrorCode SetParticlesIonAcousticWaveBuilder::build(
  const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  auto&& name = info.at("particles").get<std::string>();
  auto&& particles = simulation_.get_named_particles(name);

  PetscInt number_of_particles =
    geom_nx * geom_ny * geom_nz * particles.parameters.Np;

  PetscReal force_electric_amplitude =
    info.at("force_electric_amplitude").get<PetscReal>();

  PetscReal omega_real = info.at("omega_real").get<PetscReal>();
  PetscReal gamma = info.at("gamma").get<PetscReal>();

  Vector3R wave_number = parse_vector(info, "wave_number");
  Vector3R density_amplitude = parse_vector(info, "density_amplitude");

  Vector3R density_phase;
  if (info.contains("density_phase"))
    density_phase = parse_vector(info, "density_phase");

  Vector3R field_phase;
  if (info.contains("field_phase"))
    field_phase = parse_vector(info, "field_phase");

  commands_.emplace_back(std::make_unique<SetParticlesIonAcousticWave>(particles,
    number_of_particles, force_electric_amplitude, omega_real, gamma,
    wave_number, field_phase, density_amplitude, density_phase));

  LOG("  SetParticlesIonAcousticWave command is added for \"{}\"", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}
