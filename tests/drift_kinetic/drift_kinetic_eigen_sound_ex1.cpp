#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Ion-sound EIGENMODE test for \"drift_kinetic\" implementation.             \n"
  "A single kinetic ion-acoustic eigenmode is loaded exactly at t = 0:       \n"
  "for every species the density and the parallel bulk velocity are set to   \n"
  "  dn_s(z)/n_s = a_n_s sin(k z + phi_n_s),                                  \n"
  "  du_s(z)     = C_u_s sin(k z + phi_u_s),                                  \n"
  "with the amplitudes and phases obtained from the linear kinetic response  \n"
  "to a field E0 (script tests/drift_kinetic/ion_sound.py).  Because the      \n"
  "electron and ion amplitudes/phases differ, the two sorts are loaded by     \n"
  "SEPARATE SetParticles presets (no shared/paired coordinate).              \n";


namespace eigen {

constexpr double E0 = 3.667854e-05;

constexpr double a_n_e = 2.999886e-02;
constexpr double phi_n_e = -3.015679;
constexpr double a_n_i = 3.000000e-02;
constexpr double phi_n_i = -3.015684;

constexpr double C_u_e = 5.962568e-04;
constexpr double phi_u_e = -3.077681;
constexpr double C_u_i = 5.962796e-04;
constexpr double phi_u_i = -3.077686;

}  // namespace eigen

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
  dx = 10;
  geom_nx = 3;
  geom_x = geom_nx * dx;
  geom_ny = 3;
  geom_y = geom_ny * dx;
  geom_nz = 20;
  geom_z = geom_nz * dx;

  dt = 10;
  geom_nt = 20000;
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
        {"diagnose_period", 10*dt},
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
        {"n", +1.0},
        {"q", -1.0},
        {"m", +1.0},
        {"T", +20.0},
        {"coord_is_gc", true},
      },
      {
        {"sort_name", "ions"},
        {"Np", 5000},
        {"n", +1.0},
        {"q", +1.0},
        {"m", +100.0},
        {"T", +0.1},
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
              {"name", "SetCosineField"},
              {"min", {0.0, 0.0, 0.0}},
              {"max", {geom_x, geom_y, geom_z}},
              {"amplitude", {0.0, 0.0, eigen::E0}},
              {"wave_number", {0.0, 0.0, 1.0}},
            },
          },
        },
        {
          {"command", "SetParticles"},
          {"particles", "electrons"},
          {"coordinate", {
            {"name", "CoordinateInBoxQuietSinePaired"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"amplitude", {0.0, 0.0, eigen::a_n_e}},
            {"wave_number", {0.0, 0.0, 1.0}},
            {"phase", {0.0, 0.0, eigen::phi_n_e}},
          }},
          {"momentum", {
            {"name", "MaxwellShiftedSineQuiet"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"velocity", {0.0, 0.0, eigen::C_u_e}},
            {"wave_number", {0.0, 0.0, 1.0}},
            {"phase", {0.0, 0.0, eigen::phi_u_e}},
          }},
        },
        {
          {"command", "SetParticles"},
          {"particles", "ions"},
          {"coordinate", {
            {"name", "CoordinateInBoxQuietSinePaired"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"amplitude", {0.0, 0.0, eigen::a_n_i}},
            {"wave_number", {0.0, 0.0, 1.0}},
            {"phase", {0.0, 0.0, eigen::phi_n_i}},
          }},
          {"momentum", {
            {"name", "MaxwellShiftedSineQuiet"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"velocity", {0.0, 0.0, eigen::C_u_i}},
            {"wave_number", {0.0, 0.0, 1.0}},
            {"phase", {0.0, 0.0, eigen::phi_u_i}},
          }},
        },
      },
    },
    {
      "Diagnostics",
      {
        {
          {"diagnostic", "FieldView"},
          {"field", "E"},
          {"out_dir", "E"},
        },
        {
          {"diagnostic", "FieldView"},
          {"field", "B"},
          {"out_dir", "B"},
        },
        {
          {"diagnostic", "FieldView"},
          {"field", "electrons/J"},
          {"out_dir", "electrons/J"},
        },
        {
          {"diagnostic", "FieldView"},
          {"field", "ions/J"},
          {"out_dir", "ions/J"},
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
