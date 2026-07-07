#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Test of energy conservation for \"drift_kinetic\" implementation with "
  "paired electron-ion particles.\n";

void overwrite_config();

int main(int argc, char** argv)
{
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();
  Configuration::save(get_out_dir(__FILE__));

  drift_kinetic::Simulation simulation;
  PetscCall(simulation.initialize());
  PetscCall(simulation.calculate());
  PetscCall(simulation.finalize());

  PetscCall(PetscFinalize());
  return PETSC_SUCCESS;
}

void overwrite_config()
{
  dx = 10.;
  geom_nx = 5.;
  geom_x = geom_nx * dx;

  dt = 10.;
  geom_nt = 10000;
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", "drift_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {
      "Geometry",
      {
        {"x", geom_x},
        {"y", geom_x},
        {"z", geom_x},
        {"t", geom_t},
        {"dx", dx},
        {"dy", dx},
        {"dz", dx},
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
        {"sort_name", "electrons"},
        {"Np", 20},
        {"n", +1.0},
        {"q", -1.0},
        {"m", +1.0},
        {"T", +0.1},
        {"coord_is_gc", true},
      },
      {
        {"sort_name", "ions"},
        {"Np", 20},
        {"n", +1.0},
        {"q", +1.0},
        {"m", +100.0},
        {"T", +0.1},
        {"coord_is_gc", true},
      }},
    },
    {
      "Presets",
      {
        {
          {"command", "SetMagneticField"},
          {"field", "B0"},
          {"field_axpy", "B"},
          {
            "setter",
            {
              {"name", "SetUniformField"},
              {"value", {0.0, 0.0, 0.1}},
            },
          },
        },
        {
          {"command", "SetParticles"},
          {"particles", "electrons"},
          {"paired_with", "ions"},
          {"coordinate", {{"name", "CoordinateInBoxQuiet"}, {"min", {0.0, 0.0, 0.0}}, {"max", {30.0, 30.0, 30.0}}}},
          {"momentum", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
          {"momentum_paired", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
        },
      },
    },
  });
}
