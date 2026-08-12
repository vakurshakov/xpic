#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Ion-sound density-ringdown test for the drift_kinetic implementation at\n"
  "Te = 10 keV.  It keeps the ringdown_ex1 species ratios (Te/Ti = 200,\n"
  "mi/me = 100) and its 32-cell grid, but the longer box halves dt/dz at the\n"
  "unchanged dt, so the nonlinear push runs at vTe*dt/dz < 1.\n";

namespace ringdown {

// Twice the marker count of ringdown_ex1 per species: 3*3*32*2048.
constexpr PetscInt particles_per_cell = 2048;
constexpr PetscReal density_amplitude = 0.03;
// lambda_De = sqrt(10 keV / mec2) = 0.139891, so k*lambda_De = 0.03662.
constexpr PetscReal length_z = 24;
// dz = 0.75 gives dt/dz = 6.67 and vTe*dt/dz = 0.933, versus 12.87 and 2.55
// in ex1.
constexpr PetscInt cells_z = 32;
constexpr PetscReal time_step = 5.0;
// Kinetic root at this k: omega=3.670112e-3, gamma=2.274051e-4, T=1711.99,
// so the run covers 5.84 ion-sound periods.
constexpr PetscInt time_steps = 2000;
constexpr PetscReal diagnostic_period = 10.0;

}  // namespace ringdown

void overwrite_config();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();
  Configuration::save(get_out_dir(__FILE__));

  drift_kinetic::Simulation simulation;
  PetscCall(simulation.initialize());
  PetscCall(simulation.calculate());
  PetscCall(simulation.finalize());

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config()
{
  geom_nx = 3;
  geom_ny = 3;
  geom_nz = ringdown::cells_z;
  geom_x = 3.0;
  geom_y = 3.0;
  geom_z = ringdown::length_z;
  dx = geom_x / geom_nx;
  dy = geom_y / geom_ny;
  dz = geom_z / geom_nz;

  dt = ringdown::time_step;
  geom_nt = ringdown::time_steps;
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", "drift_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {
      "DriftKineticPush",
      {
        {"fail_on_terminal_nonconvergence", true},
      },
    },
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
        {"diagnose_period", ringdown::diagnostic_period},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "electrons"},
        {"Np", ringdown::particles_per_cell},
        {"n", +1.0},
        {"q", -1.0},
        {"m", +1.0},
        {"Tx", 0.0},
        {"Ty", 0.0},
        {"Tz", +10.0},
        {"coord_is_gc", true},
      },
      {
        {"sort_name", "ions"},
        {"Np", ringdown::particles_per_cell},
        {"n", +1.0},
        {"q", +1.0},
        {"m", +100.0},
        {"Tx", 0.0},
        {"Ty", 0.0},
        {"Tz", +0.05},
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
              {"value", {0.0, 0.0, 1.0}},
            },
          },
        },
        {
          {"command", "SetElectricField"},
          {"field", "E"},
          {
            "setter",
            {
              {"name", "SetUniformField"},
              {"value", {0.0, 0.0, 0.0}},
            },
          },
        },
        {
          {"command", "SetParticles"},
          {"particles", "electrons"},
          {"paired_with", "ions"},
          {"coordinate", {
            {"name", "CoordinateInBoxQuietSineExactPaired"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"amplitude", {0.0, 0.0, ringdown::density_amplitude}},
            {"wave_number", {0.0, 0.0, 1.0}},
            {"phase", {0.0, 0.0, 0.0}},
          }},
          {"momentum", {
            {"name", "MaxwellianVelocityQuiet"},
          }},
          {"momentum_paired", {
            {"name", "MaxwellianVelocityQuiet"},
          }},
        },
      },
    },
    {
      "Diagnostics",
      {
        {{"diagnostic", "FieldView"}, {"field", "E"}, {"component", "z"},
          {"out_dir", "E"}},
        {{"diagnostic", "DistributionMoment"}, {"particles", "electrons"},
          {"moment", "density"}, {"out_dir", "electrons/density"}},
        {{"diagnostic", "DistributionMoment"}, {"particles", "ions"},
          {"moment", "density"}, {"out_dir", "ions/density"}},
      },
    },
  });
}
