#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Analytic neutral free-streaming test for the drift_kinetic implementation.\n"
  "Equal-mass opposite-charge species start at identical positions and with\n"
  "identical velocities, so E remains zero marker by marker.\n";

namespace free_streaming {

constexpr PetscInt particles_per_cell = 128;
constexpr PetscReal density_amplitude = 0.03;
constexpr PetscReal length_z = 6.2831853071795864769;
constexpr PetscInt cells_z = 32;
constexpr PetscReal mass = 1.0;
constexpr PetscReal temperature_z = 1.0;
constexpr PetscReal time_step = 1.0;
constexpr PetscInt time_steps = 30;
constexpr PetscReal diagnostic_period = 1.0;

}  // namespace free_streaming

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
  geom_nz = free_streaming::cells_z;
  geom_x = 3.0;
  geom_y = 3.0;
  geom_z = free_streaming::length_z;
  dx = geom_x / geom_nx;
  dy = geom_y / geom_ny;
  dz = geom_z / geom_nz;

  dt = free_streaming::time_step;
  geom_nt = free_streaming::time_steps;
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", "drift_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {
      "AnalyticTest",
      {
        {"model", "neutral_free_streaming"},
        {"density_amplitude", free_streaming::density_amplitude},
        {"mode_z", 1},
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
        {"diagnose_period", free_streaming::diagnostic_period},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "negative"},
        {"Np", free_streaming::particles_per_cell},
        {"n", +1.0},
        {"q", -1.0},
        {"m", free_streaming::mass},
        {"Tx", 0.0},
        {"Ty", 0.0},
        {"Tz", free_streaming::temperature_z},
        {"coord_is_gc", true},
      },
      {
        {"sort_name", "positive"},
        {"Np", free_streaming::particles_per_cell},
        {"n", +1.0},
        {"q", +1.0},
        {"m", free_streaming::mass},
        {"Tx", 0.0},
        {"Ty", 0.0},
        {"Tz", free_streaming::temperature_z},
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
          {"particles", "negative"},
          {"paired_with", "positive"},
          {"coordinate", {
            {"name", "CoordinateInBoxQuietSineExactPaired"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"amplitude", {0.0, 0.0, free_streaming::density_amplitude}},
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
        {{"diagnostic", "DistributionMoment"}, {"particles", "negative"},
          {"moment", "density"}, {"out_dir", "negative/density"}},
        {{"diagnostic", "DistributionMoment"}, {"particles", "positive"},
          {"moment", "density"}, {"out_dir", "positive/density"}},
      },
    },
  });
}
