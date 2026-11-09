#include "src/diagnostics/distribution_moment.h"

#include "src/impls/basic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing \'DistributionMoment\' diagnostic for a sort without particles. \n"
  "`InjectParticles` runs can start with empty storage: nothing is placed  \n"
  "by the presets, but diagnostics are called at t = 0 and must write zeros.\n";

void overwrite_config();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();

  basic::Simulation simulation;
  PetscCall(simulation.initialize());

  auto& particles = *simulation.particles_[0];
  PetscInt local_size;
  PetscCall(particles.storage.get_size(&local_size));
  PetscCheck(local_size == 0, PETSC_COMM_WORLD, PETSC_ERR_USER, "The storage must be empty");

  const PetscInt dof = 6;
  Region region{
    .dim = 4,
    .dof = dof,
    .start = Vector4I{0, 0, 0, 0},
    .size = Vector4I{geom_nx, geom_ny, geom_nz, dof},
  };

  auto dir = get_out_dir(__FILE__);
  auto&& diag = DistributionMoment::create(dir, particles, moment_from_string("momentum_flux"), region);
  PetscCall(diag->diagnose(0));
  PetscCall(diag->finalize());

  std::vector<float> written;
  PetscCall(read_floats(dir / "0000", geom_nx * geom_ny * geom_nz * dof, written));
  for (float value : written)
    PetscCheck(value == 0.0f, PETSC_COMM_WORLD, PETSC_ERR_USER, "Moment of an empty sort must be zero, got %g", (double)value);

  PetscCall(simulation.finalize());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config()
{
  dx = dy = dz = 1.0;
  geom_nx = geom_ny = geom_nz = 8;
  geom_x = geom_y = geom_z = 8.0;

  dt = 1.0;
  geom_nt = 1;
  geom_t = 1.0;

  Configuration::overwrite({
    {"Simulation", "basic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {
      "Geometry",
      {
        {"x", geom_x},
        {"y", geom_y},
        {"z", geom_z},
        {"t", geom_t},
        {"dx", dx},
        {"dy", dy},
        {"dz", dz},
        {"dt", dt},
        {"diagnose_period", dt},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "particles"},
        {"Np", 1},
        {"n", 1.0},
        {"q", -1.0},
        {"m", 2.0},
      }},
    },
    {"Presets", Configuration::array_t{}},
  });
}
