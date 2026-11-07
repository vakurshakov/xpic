#include "src/impls/eccapfim/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Test of the energy and charge balance for \"eccapfim\" implementation.  \n"
  "Maxwellian electrons (ions are stationary background) are advanced for 5\n"
  "cycles in periodic boundaries. The change of the field energy must be   \n"
  "compensated by the kinetic one, dE + dB + dK = 0, and the continuity    \n"
  "equation must hold, div(J) + d(rho)/dt = 0.                             \n";

void overwrite_config();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();
  Configuration::save(get_out_dir(__FILE__));

  eccapfim::Simulation simulation;
  PetscCall(simulation.initialize());
  PetscCall(simulation.calculate());
  PetscCall(simulation.finalize());

  PetscReal max_dK, max_err, max_dQ;
  PetscCall(max_abs_column(__FILE__, "energy_conservation.txt", "dK_electrons", max_dK));
  PetscCall(max_abs_column(__FILE__, "energy_conservation.txt", "dE+dB+dK", max_err));

  PetscCall(max_abs_column(__FILE__, "charge_conservation.txt", "N2dQ_tot", max_dQ));

  PetscCheck(max_dK > 1e-6, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Kinetic energy change is too small to be checked: %g", max_dK);

  PetscCheck(max_err < 1e-5 * max_dK, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Energy balance is violated, max error: %g, max dK: %g", max_err, max_dK);

  PetscCheck(max_dQ < 1e-12, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Charge is not conserved, max error: %g", max_dQ);

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config()
{
  dx = 0.5;
  geom_nx = 10;
  geom_x = geom_nx * dx;

  dt = 1.0;
  geom_nt = 5;
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", "eccapfim"},
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
        {"Np", 10},
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
  });
}
