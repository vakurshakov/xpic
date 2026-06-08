#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Test of particle injection for \"drift_kinetic\" implementation.          \n"
  "Based on \"drift_kinetic_betta_ex1\": instead of setting the plasma cube  \n"
  "at once, electrons and ions are injected into the same cylindrical region \n"
  "with a cosine-hump radial density profile, spread uniformly over 20*dt.   \n";

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
  dx = 10.;
  geom_nx = 40;
  geom_x = geom_nx * dx;

  geom_nz = 6;
  geom_z = geom_nz * dx;

  dt = 5;
  geom_nt = 1000;
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", "drift_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {
      "Geometry",
      {
        {"x", geom_x},
        {"y", geom_x},
        {"z", geom_z},
        {"t", geom_t},
        {"dx", dx},
        {"dy", dx},
        {"dz", dx},
        {"dt", dt},
        {"diagnose_period", 4*dt},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "electrons"},
        {"Np", 5000},
        {"n", +1.5},
        {"q", -1.0},
        {"m", +1.0},
        {"T", +0.75},
        // Treat sampled `r` as the guiding center: combined with the shared
        // coordinate inside `InjectParticles`, e/i GCs coincide pair-wise.
        {"coord_is_gc", true},
      },
      {
        {"sort_name", "ions"},
        {"Np", 5000},
        {"n", +1.5},
        {"q", +1.0},
        {"m", +1.0},
        {"T", +0.75},
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
              {"value", {0.0, 0.0, 0.2}},
            },
          },
        },
      },
    },
  {
  "StepPresets",
  {
    {
      {"command", "InjectParticles"},
      {"ionized", "ions"},
      {"ejected", "electrons"},
      {"injection_start", 0},
      {"injection_end", 20 * dt},
      {"coordinate", {
        {"name", "CoordinateInCylinderCosineHump"},
        {"center", {200.0, 200.0, 30.0}},
        {"radius", 120.0},
        {"height", 60.0},
      }},
      {"momentum_i", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
      {"momentum_e", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
    },
    {
      {"command", "FieldsDamping"},
      {"damping_coefficient", 0.7},
      {"geometry", {
        {"name", "CylinderGeometry"},
        {"center", {200.0, 200.0, 30.0}},
        {"radius", 150.0},
        {"height", 60.0},
      }},
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
    {
      {"diagnostic", "FieldView"},
      {"field", "electrons/J"},
      {"out_dir", "electrons/J"},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "electrons/M"},
      {"out_dir", "electrons/M"},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "electrons/rotM"},
      {"out_dir", "electrons/rotM"},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "ions/J"},
      {"out_dir", "ions/J"},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "ions/M"},
      {"out_dir", "ions/M"},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "ions/rotM"},
      {"out_dir", "ions/rotM"},
    },
    {
      {"diagnostic", "DistributionMoment"},
      {"particles", "electrons"},
      {"moment", "density"},
      {"out_dir", "electrons/density"},
    },
    {
      {"diagnostic", "DistributionMoment"},
      {"particles", "electrons"},
      {"moment", "temperature_perp"},
      {"out_dir", "electrons/temperature_perp"},
    },
    {
      {"diagnostic", "DistributionMoment"},
      {"particles", "electrons"},
      {"moment", "temperature_parallel"},
      {"out_dir", "electrons/temperature_parallel"},
    },
    {
      {"diagnostic", "DistributionMoment"},
      {"particles", "electrons"},
      {"moment", "momentum_flux_diag_cyl"},
      {"out_dir", "electrons/momentum_flux_diag_cyl"},
    },
    {
      {"diagnostic", "DistributionMoment"},
      {"particles", "ions"},
      {"moment", "density"},
      {"out_dir", "ions/density"},
    },
    {
      {"diagnostic", "DistributionMoment"},
      {"particles", "ions"},
      {"moment", "temperature_perp"},
      {"out_dir", "ions/temperature_perp"},
    },
    {
      {"diagnostic", "DistributionMoment"},
      {"particles", "ions"},
      {"moment", "temperature_parallel"},
      {"out_dir", "ions/temperature_parallel"},
    },
    {
      {"diagnostic", "DistributionMoment"},
      {"particles", "ions"},
      {"moment", "momentum_flux_diag_cyl"},
      {"out_dir", "ions/momentum_flux_diag_cyl"},
    },
      },
    },
  });
}
