#include "src/impls/basic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Test of the Gauss law for \"basic\" implementation. Maxwellian electrons \n"
  "(T=5 keV, ions are stationary background) are advanced for 5 cycles in   \n"
  "periodic boundaries starting from E=0. The charge-conserving current must\n"
  "keep div(E) - rho = constant, i.e. div(E^5) = rho^5 - rho^0.             \n";

void overwrite_config();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();
  Configuration::save(get_out_dir(__FILE__));

  basic::Simulation simulation;
  PetscCall(simulation.initialize());

  Vec rho = simulation.particles_[0]->rho[1];

  Vec rho0, gauss;
  PetscCall(VecDuplicate(rho, &rho0));
  PetscCall(VecDuplicate(rho, &gauss));
  PetscCall(VecCopy(rho, rho0));

  PetscCall(simulation.calculate());

  PetscReal rho_max, gauss_max;
  PetscCall(VecNorm(rho, NORM_INFINITY, &rho_max));

  PetscCall(MatMult(simulation.divE, simulation.E, gauss));
  PetscCall(VecAXPY(gauss, -1.0, rho));
  PetscCall(VecAXPY(gauss, +1.0, rho0));
  PetscCall(VecNorm(gauss, NORM_INFINITY, &gauss_max));

  PetscCheck(gauss_max < 1e-10 * rho_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "div(E) - rho is not conserved, max error: %g, max rho: %g", gauss_max, rho_max);

  PetscCall(VecDestroy(&rho0));
  PetscCall(VecDestroy(&gauss));
  PetscCall(simulation.finalize());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config()
{
  dx = 0.5;
  geom_nx = 10;
  geom_x = geom_nx * dx;

  dt = 0.25;
  geom_nt = 5;
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
        {"diagnose_period", geom_t},
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
        {"T", +5.0},
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
