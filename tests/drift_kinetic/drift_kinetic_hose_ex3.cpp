#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Firehose instability test for \"drift_kinetic\" implementation.          \n"
  "Anisotropic electrons (T_par = 7.7 keV, T_perp = 770 eV, beta_par = 3,  \n"
  "beta_perp = 0.3) on top of uniform B0 = 0.1 e_z in a 20x5x20-cell box   \n"
  "with dx = 10 c/wpe. Ions are isotropic-warm with T = 770 eV and m = me. \n"
  "A cosine seed dBx = 1e-3 * cos(2 pi z / Lz) is imposed; linear theory   \n"
  "predicts Gamma = sqrt(0.35/2) * k_par * B0 with k_par = 2 pi / Lz.      \n";

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
  // dx = 10 c / wpe
  dx = 10;

  // Lx = Lz = 20 * dx; Ly is a thin transverse slab.
  geom_nx = 20;
  geom_x = geom_nx * dx;
  geom_ny = 7;
  geom_y = geom_ny * dx;
  geom_nz = 20;
  geom_z = geom_nz * dx;

  // dt = 10 / wpe = 1 / Omega_e (because Omega_e = B0 = 0.1).
  dt = 10;
  geom_nt = 2000;
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
        {"diagnose_period", 5 * dt},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "electrons"},
        {"Np", 500},
        {"n", +1.0},
        {"q", -1.0},
        {"m", +1.0},
        {"Tx", +0.77},  // T_perp = 770 eV
        {"Ty", +0.77},  // T_perp = 770 eV
        {"Tz", +0.77},   // T_parallel = 770 eV (B0 || z)
      },
      {
        {"sort_name", "ions"},
        {"Np", 500},
        {"n", +1.0},
        {"q", +1.0},
        {"m", +1.0},   // m_i = m_e
        {"T", +0.77},  // isotropic, T_i = T_perp_e = 770 eV
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
              {"value", {0.0, 0.0, 0.1}},  // B0 = 0.1 e_z
            },
          },
        },
        {
          {"command", "SetMagneticField"},
          {"field", "B"},
          {
            "setter",
            {
              {"name", "SetGeneralCosineField"},
              {"amplitude", {1.0e-3, 0.0, 0.0}},   // dBx = 1e-3
              {"wave_number", {0.0, 0.0, 1.0}},    // one wavelength along z
              {"min", {0.0, 0.0, 0.0}},
              {"max", {"geom_x", "geom_y", "geom_z"}},
            },
          },
        },
        {
          {"command", "SetElectricField"},
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
          {"coordinate", {{"name", "CoordinateInBox"}}},
          {"momentum", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
        },
        {
          {"command", "SetParticles"},
          {"particles", "ions"},
          {"coordinate", {{"name", "CoordinateInBox"}}},
          {"momentum", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
        },
      },
    },
    {
      "Diagnostics",
      {
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
          {"field", "J"},
          {"out_dir", "J_planeY"},
          {"region", {{"type", "2D"}, {"plane", "Y"}}},
        },
        {
          {"diagnostic", "FieldView"},
          {"field", "J"},
          {"out_dir", "J_planeZ"},
          {"region", {{"type", "2D"}, {"plane", "Z"}}},
        },
        {
          {"diagnostic", "FieldView"},
          {"field", "M"},
          {"out_dir", "M_planeY"},
          {"region", {{"type", "2D"}, {"plane", "Y"}}},
        },
        {
          {"diagnostic", "FieldView"},
          {"field", "M"},
          {"out_dir", "M_planeZ"},
          {"region", {{"type", "2D"}, {"plane", "Z"}}},
        },
        {
          {"diagnostic", "DistributionMoment"},
          {"particles", "electrons"},
          {"moment", "density"},
        },
        {
          {"diagnostic", "DistributionMoment"},
          {"particles", "ions"},
          {"moment", "density"},
        },
      },
    },
  });
}
