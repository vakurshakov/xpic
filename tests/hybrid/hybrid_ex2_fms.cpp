#include "src/impls/hybrid_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Fast magnetosonic wave in the hybrid_kinetic model.\n"
  "Uniform n=1, Te=Ti=1 keV, mi/me=100; Bz=B0*(1+0.05*cos(k*x)),\n"
  "B0=0.2, k=2*pi/Lx, Lx=32*dx, Ly=Lz=4*dx, dx=10, periodic boundaries.\n"
  "There are 64 markers per cell for each sort. The step dt=50 gives\n"
  "Omega_i*dt=0.1 and Omega_e*dt=10. The duration covers 2.1 periods of\n"
  "the long-wavelength FMS mode, rounded to a whole number of steps.\n"
  "B and E are written every eight steps for tools/hybrid_ex2_fms.py.\n"
  "The mass matrix is rebuilt at every field iteration; adaptive electron\n"
  "substeps handle particle iterations that need a shorter interval.\n";

namespace fms {

constexpr PetscReal B0 = 0.2;
constexpr PetscReal amplitude = 0.05;  // dBz / B0.
constexpr PetscReal ion_mass = 100.0;
constexpr PetscReal temperature = 1.0;  // [keV], both sorts.
constexpr PetscReal cell = 10.0;
constexpr PetscInt cells_x = 32;
constexpr PetscInt cells_yz = 4;
constexpr PetscReal ion_gyrofrequency = B0 / ion_mass;
constexpr PetscReal time_step = 0.1 / ion_gyrofrequency;
constexpr PetscReal wave_periods = 2.1;
constexpr PetscInt particles_per_cell = 64;
constexpr PetscReal diagnostic_period = 8.0 * time_step;

}  // namespace fms

void overwrite_config();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(PetscOptionsSetValue(nullptr, "-npc_mass_matrix_every_iteration", "true"));

  overwrite_config();
  Configuration::save(get_out_dir(__FILE__));

  hybrid_kinetic::Simulation simulation;
  PetscCall(simulation.initialize());
  PetscCall(simulation.calculate());
  PetscCall(simulation.finalize());

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config()
{
  dx = dy = dz = fms::cell;
  geom_nx = fms::cells_x;
  geom_ny = geom_nz = fms::cells_yz;
  geom_x = geom_nx * dx;
  geom_y = geom_ny * dy;
  geom_z = geom_nz * dz;

  dt = fms::time_step;
  // Long-wavelength perpendicular FMS frequency with both sorts at T;
  // cs^2 = 2*(Te+Ti)/mi, and temperatures are converted from keV.
  const PetscReal k = 2 * M_PI / geom_x;
  const PetscReal vA2 = fms::B0 * fms::B0 / fms::ion_mass;
  const PetscReal cs2 = 4 * fms::temperature / (mec2 * fms::ion_mass);
  const PetscReal omega = k * sqrt((vA2 + cs2) / (1 + vA2));
  const PetscReal period = 2 * M_PI / omega;
  geom_nt = ROUND_STEP(fms::wave_periods * period, dt);
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", "hybrid_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {"DriftKineticPush", {{"coord_is_gc", true}, {"substepping", true}}},
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
          {"Np", fms::particles_per_cell},
          {"n", +1.0},
          {"q", +1.0},
          {"m", fms::ion_mass},
          {"T", fms::temperature},
        },
        {
          {"sort_name", "electrons"},
          {"model", "drift_kinetic"},
          {"Np", fms::particles_per_cell},
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
}
