#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/random_generator.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Halton/quiet versus random loading of a uniform drift-kinetic plasma.\n";

namespace {
constexpr PetscInt particles_per_cell = 128;

PetscErrorCode run_case(bool halton, PetscReal& density_rms)
{
  PetscFunctionBeginUser;
  const PetscInt configured_particles = halton ? particles_per_cell / 2 : particles_per_cell;
  Configuration::json_t presets = {
    {{"command", "SetField"}, {"field", "B"},
      {"setter", {{"name", "SetUniformField"}, {"value", {0, 0, 1}}}}},
  };
  if (halton) {
    for (const char* sort : {"ions", "electrons"})
      presets.push_back({
        {"command", "SetParticlesIonAcousticWave"}, {"particles", sort},
        {"wave_number", {0, 0, 1}}, {"density_amplitude", {0, 0, 0}},
        {"force_electric_amplitude", 0}, {"omega_real", 0.01}, {"gamma", 0.01},
      });
  }
  else {
    presets.push_back({
      {"command", "SetParticles"}, {"particles", "ions"}, {"paired_with", "electrons"},
      {"coordinate", {{"name", "CoordinateInBox"}}},
      {"momentum", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
      {"momentum_paired", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
    });
  }
  Configuration::overwrite({
    {"Simulation", "drift_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {"DriftKineticPush", {{"coord_is_gc", true}}},
    {"FieldPreconditioner", "none"},
    {"Geometry",
      {
        {"x", 40}, {"y", 40}, {"z", 40}, {"t", 100},
        {"dx", 10}, {"dy", 10}, {"dz", 10}, {"dt", 10},
        {"diagnose_period", 10},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      }},
    {"Particles",
      {
        {{"sort_name", "ions"}, {"model", "drift_kinetic"},
          {"Np", configured_particles}, {"n", 1}, {"q", 1}, {"m", 100}, {"T", 0.1}},
        {{"sort_name", "electrons"}, {"model", "drift_kinetic"},
          {"Np", configured_particles}, {"n", 1}, {"q", -1}, {"m", 1}, {"T", 0.1}},
      }},
    {"Presets", presets},
  });
  RandomGenerator::get().seed(1729);
  drift_kinetic::Simulation simulation;
  PetscCall(simulation.initialize());
  auto& electrons = simulation.get_named_particles("electrons");
  // The IAW preset loads Np pairs; correct its weight for 2*Np particles.
  if (halton)
    PetscCall(VecScale(electrons.rho[1], 0.5));
  PetscCall(VecShift(electrons.rho[1], 1));
  PetscCall(VecNorm(electrons.rho[1], NORM_2, &density_rms));
  density_rms /= std::sqrt(geom_nx * geom_ny * geom_nz);
  PetscCall(simulation.finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode run()
{
  PetscFunctionBeginUser;
  PetscReal halton_density, random_density;
  PetscCall(run_case(true, halton_density));
  PetscCall(run_case(false, random_density));

  PetscCheck(halton_density < random_density, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Initial Halton density noise %g must be below random %g", halton_density, random_density);

  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(run());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
