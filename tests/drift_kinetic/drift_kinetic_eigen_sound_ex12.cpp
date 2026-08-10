#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Ion-sound kinetic-loading test for the drift_kinetic implementation.     \n"
  "The density of each species is the exact configured sine.  At every z,  \n"
  "v_parallel is sampled from the positive regularization of the linear     \n"
  "kinetic response to E_z = E0 cos(kz), rather than from a locally shifted \n"
  "Maxwellian.  The negative part of the linearized PDF is clipped and the  \n"
  "conditional velocity distribution is normalized independently at z.     \n";

namespace eigen {

constexpr double E0 = 3.667854e-05;
constexpr double omega_real = 6.232227e-04;
constexpr double gamma = 3.869059e-05;

constexpr double a_n_e = 2.999886e-02;
constexpr double phi_n_e = -3.015679;
constexpr double a_n_i = 3.000000e-02;
constexpr double phi_n_i = -3.015684;

constexpr double vpar_max_e = 0.95;
constexpr double vpar_max_i = 1.1191274207850656e-02;
constexpr double mu_max_e = 4.6966731898238745e-01;
constexpr double mu_max_i = 2.3483365949119378e-03;

}  // namespace eigen

void overwrite_config();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();
  std::filesystem::create_directories(get_out_dir(__FILE__));
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
        {"diagnose_period", 10 * dt},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "electrons"},
        {"Np", 10000},
        {"n", +1.0},
        {"q", -1.0},
        {"m", +1.0},
        {"T", +20.0},
        {"coord_is_gc", true},
      },
      {
        {"sort_name", "ions"},
        {"Np", 10000},
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
            {"name", "CoordinateInBoxQuietSineExactPaired"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"amplitude", {0.0, 0.0, eigen::a_n_e}},
            {"wave_number", {0.0, 0.0, 1.0}},
            {"phase", {0.0, 0.0, eigen::phi_n_e}},
          }},
          {"momentum", {
            {"name", "KineticIonSoundQuiet"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"electric_amplitude", eigen::E0},
            {"omega_real", eigen::omega_real},
            {"gamma", eigen::gamma},
            {"wave_number", {0.0, 0.0, 1.0}},
            {"field_phase", {0.0, 0.0, 0.0}},
            {"velocity_cutoff_vT", 8.0},
            {"velocity_abs_max", eigen::vpar_max_e},
          }},
        },
        {
          {"command", "SetParticles"},
          {"particles", "ions"},
          {"coordinate", {
            {"name", "CoordinateInBoxQuietSineExactPaired"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"amplitude", {0.0, 0.0, eigen::a_n_i}},
            {"wave_number", {0.0, 0.0, 1.0}},
            {"phase", {0.0, 0.0, eigen::phi_n_i}},
          }},
          {"momentum", {
            {"name", "KineticIonSoundQuiet"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"electric_amplitude", eigen::E0},
            {"omega_real", eigen::omega_real},
            {"gamma", eigen::gamma},
            {"wave_number", {0.0, 0.0, 1.0}},
            {"field_phase", {0.0, 0.0, 0.0}},
            {"velocity_cutoff_vT", 8.0},
            {"velocity_abs_max", 0.95},
          }},
        },
      },
    },
    {
      "Diagnostics",
      {
        {{"diagnostic", "FieldView"}, {"field", "E"}, {"out_dir", "E"}},
        {{"diagnostic", "FieldView"}, {"field", "B"}, {"out_dir", "B"}},
        {{"diagnostic", "FieldView"}, {"field", "electrons/J"},
          {"out_dir", "electrons/J"}},
        {{"diagnostic", "FieldView"}, {"field", "electrons/rotM"},
          {"out_dir", "electrons/rotM"}},
        {{"diagnostic", "FieldView"}, {"field", "ions/J"},
          {"out_dir", "ions/J"}},
        {{"diagnostic", "FieldView"}, {"field", "ions/rotM"},
          {"out_dir", "ions/rotM"}},
        {{"diagnostic", "DistributionMoment"}, {"particles", "electrons"},
          {"moment", "density"}},
        {{"diagnostic", "DistributionMoment"}, {"particles", "electrons"},
          {"moment", "temperature_parallel"}},
        {{"diagnostic", "DistributionMoment"}, {"particles", "electrons"},
          {"moment", "temperature_perp"}},
        {{"diagnostic", "DistributionMoment"}, {"particles", "ions"},
          {"moment", "density"}},
        {{"diagnostic", "DistributionMoment"}, {"particles", "ions"},
          {"moment", "temperature_parallel"}},
        {{"diagnostic", "DistributionMoment"}, {"particles", "ions"},
          {"moment", "temperature_perp"}},
        {
          {"diagnostic", "DkDistributionFunction"},
          {"particles", "electrons"},
          {"out_dir", "electrons/distribution_function"},
          {"v_parallel", {
            {"min", -eigen::vpar_max_e}, {"max", eigen::vpar_max_e},
            {"bins", 3000}}},
          {"mu_p", {
            {"min", 0.0}, {"max", eigen::mu_max_e}, {"bins", 48}}},
          {"diagnose_period", 10 * dt},
          {"max_frames", 5},
        },
        {
          {"diagnostic", "DkDistributionFunction"},
          {"particles", "ions"},
          {"out_dir", "ions/distribution_function"},
          {"v_parallel", {
            {"min", -eigen::vpar_max_i}, {"max", eigen::vpar_max_i},
            {"bins", 192}}},
          {"mu_p", {
            {"min", 0.0}, {"max", eigen::mu_max_i}, {"bins", 48}}},
          {"diagnose_period", 10 * dt},
          {"max_frames", 5},
        },
      },
    },
  });
}
