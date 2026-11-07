#include "src/impls/eccapfim/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Test of energy and charge conservation for \"eccapfim\" implementation. \n"
  "The simplest case is tested: plasma cube of size L=5.0 (N=10) is modeled\n"
  "in periodic boundaries for 100 cycles (dt=1.5). There are only          \n"
  "maxwellian electrons with the temperature T=100 eV, ions are stationary \n"
  "background.                                                             \n";

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

  // The total energy W = wE + wB + wK is conserved up to the solver tolerance:
  // the running sum of per-step balances dE + dB + dK is exactly W(t) - W(0).
  PetscReal max_wK, max_drift;
  PetscCall(max_abs_column(__FILE__, "energy.txt", "wK_electrons", max_wK));
  PetscCall(max_abs_column(__FILE__, "energy_conservation.txt", "dE+dB+dK", max_drift, true));

  PetscCheck(max_drift < 1e-6 * max_wK, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Total energy is not conserved, max drift: %g, max wK: %g", max_drift, max_wK);

  // The trajectory is kept in check by the peaks of non-conserved quantities,
  // which are bounded from above by about twice the values of the validated run
  PetscReal max_wE, max_dK;
  PetscCall(max_abs_column(__FILE__, "energy.txt", "wE", max_wE));
  PetscCall(max_abs_column(__FILE__, "energy_conservation.txt", "dK_electrons", max_dK));

  PetscCheck(max_wE < 1e-2, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Peak of the electric energy is too large: %g", max_wE);

  PetscCheck(max_dK < 1e-2, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Peak of the kinetic energy change per timestep is too large: %g", max_dK);

  // Charge is conserved by the scheme, compared to "ecsim"
  PetscReal max_dQ;
  PetscCall(max_abs_column(__FILE__, "charge_conservation.txt", "N2dQ_tot", max_dQ));

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

  dt = 1.5;
  geom_nt = 100;
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
        {"diagnose_period", geom_nt / 2},
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
