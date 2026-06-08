#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Test of energy and charge conservation for \"drift_kinetic\" implementation.  \n"
  "The simplest case is tested: plasma cube of size L=5.0 (N=10) is modeled \n"
  "in periodic boundaries for 100 cycles (dt=1.5). There are only maxwellian\n"
  "electrons with the temperature T=100 eV, ions are stationary background. \n";

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

  dt = 10;
  geom_nt = 2;
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
        {"diagnose_period", dt},
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
        {"n", +1.0},
        {"q", -1.0},
        {"m", +1.0},
        {"T", +0.75},
      },
      {
        {"sort_name", "ions"},
        {"Np", 1000},
        {"n", +1.0},
        {"q", +1.0},
        {"m", +1836.0},
        {"T", +0.75},
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
          {"coordinate", {
            {"name", "CoordinateInCylinderCosineHump"},
            {"center", {200.0, 200.0, 30.0}},
            {"radius", 120.0},
            {"height", 60.0}
          }},
          {"momentum", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
        },
        {
          {"command", "SetParticles"},
          {"particles", "ions"},
          {"coordinate", {
            {"name", "CoordinateInCylinderCosineHump"},
            {"center", {200.0, 200.0, 30.0}},
            {"radius", 120.0},
            {"height", 60.0}
          }},
          {"momentum", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
        },
      },
    },
  {
  "StepPresets",
  {
    {
      {"command", "FieldsDamping"},
      {"damping_coefficient", 0.35},
      {"geometry", {
        {"name", "CylinderGeometry"},
        {"center", {200.0, 200.0, 30.0}},
        {"radius", 170.0},
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
      {"out_dir", "B_planeY"},
      {"region", {{"type", "2D"}, {"plane", "Y"}}},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "B"},
      {"out_dir", "B_planeZ"},
      {"region", {{"type", "2D"}, {"plane", "Z"}}},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "E"},
      {"out_dir", "E_planeY"},
      {"region", {{"type", "2D"}, {"plane", "Y"}}},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "E"},
      {"out_dir", "E_planeZ"},
      {"region", {{"type", "2D"}, {"plane", "Z"}}},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "J"},
      {"out_dir", "J_planeZ"},
      {"region", {{"type", "2D"}, {"plane", "Z"}}},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "J"},
      {"out_dir", "J_planeY"},
      {"region", {{"type", "2D"}, {"plane", "Y"}}},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "M"},
      {"out_dir", "M_planeZ"},
      {"region", {{"type", "2D"}, {"plane", "Z"}}},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "M"},
      {"out_dir", "M_planeY"},
      {"region", {{"type", "2D"}, {"plane", "Y"}}},
    },
    {
      {"diagnostic", "DistributionMoment"},
      {"particles", "electrons"},
      {"moment", "density"},
      {"out_dir", "electrons/density"},
    },
    {
      {"diagnostic", "DistributionMoment"},
      {"particles", "ions"},
      {"moment", "density"},
      {"out_dir", "ions/density"},
    },
      },
    },
  });
}
