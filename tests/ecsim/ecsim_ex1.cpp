#include "src/impls/ecsim/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static char help[] =
  "Test of energy conservation for \"ecsim\" implementation. \n"
  "The simplest case is tested: plasma cube of size L=5.0 (N=10) is modeled \n"
  "in periodic boundaries for 100 cycles (dt=1.5). There are only maxwellian\n"
  "electrons with the temperature T=100 eV, ions are stationary background. \n";

/// @todo Create a binary files comparator to be able to tell where is the difference

void overwrite_config();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();

  ecsim::Simulation simulation;
  PetscCall(simulation.initialize());
  PetscCall(simulation.calculate());
  PetscCall(simulation.finalize());

  PetscCall(compare_temporal(__FILE__, "energy_conservation.txt"));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config()
{
  dx = 0.5;
  geom_nx = 10;
  geom_x = geom_nx * dx;

  dt = 1.5;
  geom_nt = 100;
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", "ecsim"},
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
        {"diagnose_period", geom_t / 2},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "electrons"},
        {"Np", 100},
        {"n", +1.0},
        {"q", -1.0},
        {"m", +1.0},
        {"T", +0.1},
      }},
    },
    {
      "Presets",
      {{
        {"command", "SetParticles"},
        {"particles", "electrons"},
        {"coordinate", {{"name", "CoordinateInBox"}}},
        {"momentum", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
      }},
    },
    {
      "Diagnostics",
      {
        {{"diagnostic", "FieldView"}, {"field", "E"}},
        {{"diagnostic", "FieldView"}, {"field", "B"}},
        {
          {"diagnostic", "DistributionMoment"},
          {"particles", "electrons"},
          {"moment", "Density"},
        },
      },
    },
  });
}
