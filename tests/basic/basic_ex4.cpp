#include "src/impls/basic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Test of energy and charge conservation for \"basic\" implementation.      \n"
  "The simplest case is tested: plasma cube of size L=0.5 (N=10) is modeled  \n"
  "in periodic boundaries for 100 cycles (dt=0.25). There are only maxwellian\n"
  "electrons with the temperature T=100 eV, ions are stationary background. \n";

void overwrite_config();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();
  Configuration::save(get_out_dir(__FILE__));

  basic::Simulation simulation;
  PetscCall(simulation.initialize());
  PetscCall(simulation.calculate());
  PetscCall(simulation.finalize());

  // The explicit scheme does not conserve energy exactly, so the drift of the
  // total energy W = wE + wB + wK (running sum of per-step balances dE + dB + dK
  // is exactly W(t) - W(0)) is bounded, not compared to round-off.
  PetscReal max_wK, max_drift;
  PetscCall(max_abs_column(__FILE__, "energy.txt", "wK_electrons", max_wK));
  PetscCall(max_abs_column(__FILE__, "energy_conservation.txt", "dE+dB+dK", max_drift, true));

  PetscCheck(max_drift < 1e-3 * max_wK, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Total energy drift is too large, max drift: %g, max wK: %g", max_drift, max_wK);

  // The trajectory is kept in check by the peaks of non-conserved quantities,
  // which are bounded from above by about twice the values of the validated run
  PetscReal max_wE, max_dK;
  PetscCall(max_abs_column(__FILE__, "energy.txt", "wE", max_wE));
  PetscCall(max_abs_column(__FILE__, "energy_conservation.txt", "dK_electrons", max_dK));

  PetscCheck(max_wE < 1e-3, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Peak of the electric energy is too large: %g", max_wE);

  PetscCheck(max_dK < 1e-5, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Peak of the kinetic energy change per timestep is too large: %g", max_dK);

  // The residual errors of the order ~1e-12 are round-off noise that varies
  // between runs (atomic updates order), so only its magnitude is checked.
  PetscReal max_dQ;
  PetscCall(max_abs_column(__FILE__, "charge_conservation.txt", "N2dQ_tot", max_dQ));

  PetscCheck(max_dQ < 1e-11, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Charge is not conserved, max error: %g", max_dQ);

  // The balance dP/dt = sum[q (E + v x B)] holds to round-off
  PetscReal max_dP;
  PetscCall(max_abs_column(__FILE__, "momentum_conservation.txt", "N2dP", max_dP));

  PetscCheck(max_dP < 1e-13, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Momentum balance is violated, max error: %g", max_dP);

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config()
{
  dx = 0.05;
  geom_nx = 10;
  geom_x = geom_nx * dx;

  dt = 0.025;
  geom_nt = 100;
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", "basic"},
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
  });
}
