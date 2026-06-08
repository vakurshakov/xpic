#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Test of direct (no injection) particle loading for \"drift_kinetic\".     \n"
  "Based on \"drift_kinetic_beta_ex3\": instead of `InjectParticles`, the    \n"
  "electron/ion pair is loaded at once via `SetParticles` + `paired_with`,   \n"
  "with `CoordinateInBoxCosineHump` (axis = X). The domain is a long, narrow\n"
  "box (geom_nx = 60, geom_ny = geom_nz = 3) periodic on all three axes.    \n"
  "FieldsDamping uses a CylinderGeometry whose axis is Z, so the radius is  \n"
  "measured in the X-Y plane and the damper is effectively 1D along X,      \n"
  "symmetric about the plasma center x = 300. Fields are left untouched for \n"
  "r < radius (x in ~[150, 450]) and damped outside. The coefficient is     \n"
  "intentionally small (0.15) so the layer turns on gradually: a soft       \n"
  "impedance gradient absorbs the X-propagating wave with little reflection,\n"
  "applied every step over the whole band, instead of the abrupt x0.3-per- \n"
  "step wall that a large coefficient produces. NOTE: with this formula the \n"
  "very domain boundary (r ~ center) stays transparent, so the periodic     \n"
  "wrap-around relies on the band absorbing the wave before it reaches the  \n"
  "edge -- hence the wide, gentle layer.                                    \n";

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
  geom_nx = 60;
  geom_x = geom_nx * dx;

  geom_ny = 3;
  geom_y = geom_ny * dx;

  geom_nz = 3;
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
        {"y", geom_y},
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
        {"Np", 1000},
        {"n", +1.5},
        {"q", -1.0},
        {"m", +1.0},
        {"T", +0.75},
        {"coord_is_gc", true},
      },
      {
        {"sort_name", "ions"},
        {"Np", 1000},
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
        {
          {"command", "SetParticles"},
          {"particles", "electrons"},
          {"paired_with", "ions"},
          {"coordinate", {
            {"name", "CoordinateInBoxCosineHump"},
            {"axis", "X"},
            {"min", {180.0, 0.0, 0.0}},
            {"max", {420.0, geom_y, geom_z}},
          }},
          {"momentum",        {{"name", "MaxwellianMomentum"}, {"tov", true}}},
          {"momentum_paired", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
        },
      },
    },
  {
  "StepPresets",
  {
    {
      {"command", "FieldsDamping"},
      // Small coefficient => gentle onset at the inner edge of the layer, so
      // the X-propagating wave is absorbed gradually instead of reflecting off
      // an abrupt damping wall. The damper is applied every step, so even a
      // weak per-step factor accumulates to strong absorption across the band.
      {"damping_coefficient", 0.15},
      {"geometry", {
        {"name", "CylinderGeometry"},
        // Centered on the domain (x = 300) => damper is symmetric in x.
        {"center", {300.0, 15.0, 15.0}},
        // radius = 150 keeps the no-damp region x in ~[150, 450] (the cosine
        // hump lives in [180, 420], so it is untouched) while giving a wide
        // 150-wide gradient toward each periodic boundary.
        {"radius", 150.0},
        {"height", geom_z},
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
