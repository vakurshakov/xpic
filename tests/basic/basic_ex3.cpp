#include "src/impls/basic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Test of the momentum balance for \"basic\" implementation. Maxwellian    \n"
  "electrons (ions are stationary background) are advanced for 5 cycles in  \n"
  "the static non-uniform E and B fields. The momentum diagnostic must      \n"
  "report (P^{n+1} - P^n) / dt = sum[q * (E + v x B)] up to round-off.      \n";

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

  // The force on particles: max(QE[c], QVB[c])
  PetscReal max_f = 0.0;
  for (const char* title : {                                 //
         "QEx_electrons", "QEy_electrons", "QEz_electrons",  //
         "QVBx_electron", "QVBy_electron", "QVBz_electron"}) {
    PetscReal max_c;
    PetscCall(max_abs_column(__FILE__, "momentum_conservation.txt", title, max_c));
    max_f = std::max(max_f, max_c);
  }

  // There is no step to balance at t=0
  PetscReal max_err;
  PetscCall(max_abs_column(__FILE__, "momentum_conservation.txt", "N2dP", max_err, false, 1));

  PetscCheck(max_f > 1e-6, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Force on particles is too small to be checked: %g", max_f);

  PetscCheck(max_err < 1e-12 * max_f, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Momentum balance is violated, max error: %g, max force: %g", max_err, max_f);

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
    {"Fields", {{"evolve", false}}},
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
        {"T", +0.1},
      }},
    },
    {
      "Presets",
      {
        {
          {"command", "SetField"},
          {"field", "B"},
          {"setter",
            {
              {"name", "SetGeneralCosineField"},
              {"amplitude", {0.3, -0.2, 0.4}},
              {"wave_number", {1.0, 1.0, 2.0}},
            }},
        },
        {
          {"command", "SetField"},
          {"field", "E"},
          {"setter",
            {
              {"name", "SetGeneralCosineField"},
              {"amplitude", {0.01, 0.02, 0.03}},
              {"wave_number", {1.0, 2.0, 1.0}},
            }},
        },
        {
          {"command", "SetParticles"},
          {"particles", "electrons"},
          {"coordinate", {{"name", "CoordinateInBox"}}},
          {"momentum", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
        },
      },
    },
  });
}
