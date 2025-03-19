#include "src/impls/ecsimcorr/simulation.h"
#include "src/utils/configuration.h"
#include "src/utils/world.h"
#include "tests/common.h"

static char help[] =
  "Test of energy and charge conservation for ecsimcorr implementation.\n"
  "Cube of size L=pi [c/w_pe] (N=32) is modeled in periodic boundaries\n"
  "for 1000 cycles (dt=0.1 [1/w_pe]).\n";

void overwrite_config();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  /// @todo Create `interfaces::Simulation::finalize()`, where we can destroy all petsc objects
  {
    overwrite_config();

    ecsimcorr::Simulation simulation;
    PetscCall(simulation.initialize());
    PetscCall(simulation.calculate());
  }

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config()
{
  static constexpr PetscReal size_cwpe = M_PI;
  static constexpr PetscInt size_cell = 32;

  static constexpr PetscReal cell_size =
    size_cwpe / static_cast<PetscReal>(size_cell);

  Configuration::overwrite({
    {"OutputDirectory", get_out_dir(__FILE__)},
    {
      "Geometry",
      {
        {"x", size_cwpe},
        {"y", size_cwpe},
        {"z", size_cwpe},
        {"t", 3.0},
        {"dx", cell_size},
        {"dy", cell_size},
        {"dz", cell_size},
        {"dt", 0.1},
        {"diagnose_period", 1.5},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      /// @note Ions are stationary background in this example
      "Particles",
      {{
        {"sort_name", "electrons"},
        {"Np", 10},
        {"n", +1.0},
        {"q", -1.0},
        {"m", +1.0},
        {"T", +1.0},
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
      /// @todo Create a binary files comparator to be able to tell where is the difference
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
