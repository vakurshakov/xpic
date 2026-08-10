#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Fast, resolved long-wavelength ion-sound density ringdown.\n"
  "Compared with ringdown_ex5 it halves the wavelength, resolves it with\n"
  "32 cells and keeps nearly the same total marker count per species.\n";

namespace ringdown {

constexpr PetscInt particles_per_cell = 3072;
constexpr PetscReal density_amplitude = 0.03;
constexpr PetscReal length_z = 100.0;
constexpr PetscInt cells_z = 32;
constexpr PetscReal time_step = 10.0;
// Five ion-sound periods: omega_th=1.24638e-3 and T=5041.16.
constexpr PetscInt time_steps = 2521;
constexpr PetscReal diagnostic_period = 50.0;

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
  geom_x = 30.0;
  geom_y = 30.0;
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
        {"Tz", +20.0},
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
        {"Tz", +0.1},
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
