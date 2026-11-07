#include "src/impls/hybrid_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Fast magnetosonic wave in the \"hybrid_kinetic\" model.                  \n"
  "                                                                          \n"
  "A uniform Maxwellian plasma n = 1, Te = Ti = 1 keV, mi/me = 100 is placed \n"
  "in the field Bz(x) = B0 (1 + 0.05 cos(k x)), B0 = 0.2, k = 2 pi / Lx,     \n"
  "Lx = 32 dx, Ly = Lz = 3 dx, dx = 10, dt = 25, periodic boundaries. Ions   \n"
  "are fully kinetic (Crank-Nicolson), electrons are drift-kinetic guiding   \n"
  "centers (Omega_e dt = 5). The perturbation excites a standing fast        \n"
  "magnetosonic wave, omega = k sqrt((vA^2 + cs^2) / (1 + vA^2)) = 0.215     \n"
  "Omega_i, on top of a static pressure-balanced structure. B and E are      \n"
  "written every eight steps and compared with the theory by                 \n"
  "tools/hybrid_ex2_fms.py. The options `-npc` and `-time` override   \n"
  "the markers per cell and the simulated time.                              \n"
  "                                                                          \n"
  "The field iterations are preconditioned by the mass matrix, rebuilt at    \n"
  "every iteration: the E x B response of the electrons is stiff,            \n"
  "(dt / 2) omega_pe^2 / Omega_e = 62, and a matrix of the first iterate     \n"
  "slows the convergence down from 7 to 11-14 iterations per step. The step  \n"
  "is limited by the Picard push of the electrons, whose contraction grows   \n"
  "as dt^2 (q/m) dE_par/ds: at dt = 50 some of them already need halving.    "
  "\n";

namespace fms {

constexpr PetscReal B0 = 0.2;
constexpr PetscReal amplitude = 0.05;  // dBz / B0.
constexpr PetscReal ion_mass = 100.0;
constexpr PetscReal temperature = 1.0;  // [keV], both sorts.
constexpr PetscReal cell = 10.0;
constexpr PetscInt cells_x = 32;
constexpr PetscInt cells_yz = 3;
constexpr PetscReal time_step = 25.0;
// The wave period is 2 pi / omega = 1.46e4, about 585 steps; four periods
// separate the static part from the wave. With fewer than 128 markers per
// cell the noise shifts the fitted static part by 0.04-0.07 eps B0.
constexpr PetscReal total_time = 6e4;
constexpr PetscInt particles_per_cell = 128;
constexpr PetscReal diagnostic_period = 8.0 * time_step;

}  // namespace fms

PetscErrorCode overwrite_config();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(PetscOptionsSetValue(nullptr, "-npc_mass_matrix_every_iteration", "true"));

  PetscCall(overwrite_config());
  Configuration::save(get_out_dir(__FILE__));

  hybrid_kinetic::Simulation simulation;
  PetscCall(simulation.initialize());
  PetscCall(simulation.calculate());
  PetscCall(simulation.finalize());

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode overwrite_config()
{
  PetscFunctionBeginUser;
  PetscReal time = fms::total_time;
  PetscInt npc = fms::particles_per_cell;
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-time", &time, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-npc", &npc, nullptr));

  dx = dy = dz = fms::cell;
  geom_nx = fms::cells_x;
  geom_ny = geom_nz = fms::cells_yz;
  geom_x = geom_nx * dx;
  geom_y = geom_ny * dy;
  geom_z = geom_nz * dz;

  dt = fms::time_step;
  geom_nt = ROUND_STEP(time, dt);
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", "hybrid_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {"DriftKineticPush", {{"coord_is_gc", true}}},
    {"FieldPreconditioner", "mass_matrix"},
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
        {"diagnose_period", fms::diagnostic_period},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {
        {
          {"sort_name", "ions"},
          {"model", "kinetic"},
          {"Np", npc},
          {"n", +1.0},
          {"q", +1.0},
          {"m", fms::ion_mass},
          {"T", fms::temperature},
        },
        {
          {"sort_name", "electrons"},
          {"model", "drift_kinetic"},
          {"Np", npc},
          {"n", +1.0},
          {"q", -1.0},
          {"m", +1.0},
          {"T", fms::temperature},
        },
      },
    },
    {
      "Presets",
      {
        {
          {"command", "SetField"},
          {"field", "B"},
          {"setter",
            {
              {"name", "SetUniformField"},
              {"value", {0.0, 0.0, fms::B0}},
            }},
        },
        {
          {"command", "SetField"},
          {"field", "B"},
          {"setter",
            {
              {"name", "SetGeneralCosineField"},
              {"amplitude", {0.0, 0.0, fms::B0 * fms::amplitude}},
              {"wave_number", {1.0, 0.0, 0.0}},
            }},
        },
        // Set after B: the magnetic moments of the guiding centers use the
        // local field. Guiding centers are placed at the ion positions, so the
        // initial charge density is zero.
        {
          {"command", "SetParticles"},
          {"particles", "ions"},
          {"paired_with", "electrons"},
          {"coordinate", {{"name", "CoordinateInBox"}}},
          {"momentum", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
          {"momentum_paired", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
        },
      },
    },
    {
      "Diagnostics",
      {
        {
          {"diagnostic", "FieldView"},
          {"field", "B"},
          {"out_dir", "B"},
        },
        {
          {"diagnostic", "FieldView"},
          {"field", "E"},
          {"out_dir", "E"},
        },
      },
    },
  });
  PetscFunctionReturn(PETSC_SUCCESS);
}
