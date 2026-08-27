#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Travelling ion-sound twin of drift_kinetic_ringdown_ex15.              \n"
  "As in ringdown_ex10, E and the first three kinetic moments of each      \n"
  "species are initialized from the full-Z Landau root, selecting the      \n"
  "forward quasimode and suppressing the counter-propagating branch.       \n";

namespace travelling_wave {

constexpr double theory_mec2_kev = 511.0;
constexpr double theory_Te_kev = 20.0;
constexpr double theory_Ti_kev = 0.1;
constexpr double theory_ne = 1.0;
constexpr double theory_ni = 1.0;
constexpr double theory_qe = -1.0;
constexpr double theory_qi = 1.0;
constexpr double theory_me = 1.0;
constexpr double theory_mi = 100.0;
constexpr double theory_Lz = 200.0;
constexpr double theory_dz = 10.0;
constexpr double theory_mode = 1.0;
constexpr double theory_dn_i = 0.03;

constexpr double E_force = 3.6678438440940783e-05;
constexpr double E_grid = 3.6981601028325305e-05;
constexpr double omega_real = 6.2322182890254662e-04;
constexpr double gamma = 3.8690499431800553e-05;
constexpr double a_n_e = 2.9998837859139024e-02;
constexpr double phi_n_e = -3.0156794697524187e+00;
constexpr double a_n_i = 2.9999999999999999e-02;
constexpr double phi_n_i = -3.0156843733207972e+00;
constexpr double field_grid_phase = 1.5707963267948966e-01;

constexpr double vpar_max_e = 0.95;
constexpr double vpar_max_i = 1.1191274207850656e-02;
constexpr double mu_max_e = 9.3933437978437490e-01;
constexpr double mu_max_i = 2.3483365949119378e-03;

constexpr PetscInt particles_per_cell = 15000;
constexpr PetscInt cells_z = static_cast<PetscInt>(theory_Lz / theory_dz);
constexpr PetscReal time_step = 10.0;
constexpr PetscInt time_steps = 20000;
constexpr PetscReal diagnostic_period = 100.0;

}  // namespace travelling_wave

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
  geom_nx = 3;
  geom_ny = 3;
  geom_nz = travelling_wave::cells_z;
  dx = travelling_wave::theory_dz;
  dy = travelling_wave::theory_dz;
  dz = travelling_wave::theory_dz;
  geom_x = geom_nx * dx;
  geom_y = geom_ny * dy;
  geom_z = geom_nz * dz;

  dt = travelling_wave::time_step;
  geom_nt = travelling_wave::time_steps;
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
        {"diagnose_period", travelling_wave::diagnostic_period},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "electrons"},
        {"Np", travelling_wave::particles_per_cell},
        {"n", travelling_wave::theory_ne},
        {"q", travelling_wave::theory_qe},
        {"m", travelling_wave::theory_me},
        {"Tx", 0.0},
        {"Ty", 0.0},
        {"Tz", travelling_wave::theory_Te_kev},
        {"coord_is_gc", true},
      },
      {
        {"sort_name", "ions"},
        {"Np", travelling_wave::particles_per_cell},
        {"n", travelling_wave::theory_ni},
        {"q", travelling_wave::theory_qi},
        {"m", travelling_wave::theory_mi},
        {"Tx", 0.0},
        {"Ty", 0.0},
        {"Tz", travelling_wave::theory_Ti_kev},
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
          {"setter", {{"name", "SetUniformField"},
                      {"value", {0.0, 0.0, 1.0}}}},
        },
        {
          {"command", "SetElectricField"},
          {"field", "E"},
          {"setter", {
            {"name", "SetCosineField"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"amplitude", {0.0, 0.0, travelling_wave::E_grid}},
            {"wave_number", {0.0, 0.0, travelling_wave::theory_mode}},
            {"phase", {0.0, 0.0, travelling_wave::field_grid_phase}},
          }},
        },
        {
          {"command", "SetParticles"},
          {"particles", "electrons"},
          {"coordinate", {
            {"name", "CoordinateInBoxQuietSineExactLatticePaired"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"amplitude", {0.0, 0.0, travelling_wave::a_n_e}},
            {"wave_number", {0.0, 0.0, travelling_wave::theory_mode}},
            {"phase", {0.0, 0.0, travelling_wave::phi_n_e}},
          }},
          {"momentum", {
            {"name", "KineticIonSoundMomentsQuiet"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"force_electric_amplitude", travelling_wave::E_force},
            {"omega_real", travelling_wave::omega_real},
            {"gamma", travelling_wave::gamma},
            {"wave_number", {0.0, 0.0, travelling_wave::theory_mode}},
            {"field_phase", {0.0, 0.0, 0.0}},
            {"density_amplitude", {0.0, 0.0, travelling_wave::a_n_e}},
            {"density_phase", {0.0, 0.0, travelling_wave::phi_n_e}},
          }},
        },
        {
          {"command", "SetParticles"},
          {"particles", "ions"},
          {"coordinate", {
            {"name", "CoordinateInBoxQuietSineExactLatticePaired"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"amplitude", {0.0, 0.0, travelling_wave::a_n_i}},
            {"wave_number", {0.0, 0.0, travelling_wave::theory_mode}},
            {"phase", {0.0, 0.0, travelling_wave::phi_n_i}},
          }},
          {"momentum", {
            {"name", "KineticIonSoundMomentsQuiet"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"force_electric_amplitude", travelling_wave::E_force},
            {"omega_real", travelling_wave::omega_real},
            {"gamma", travelling_wave::gamma},
            {"wave_number", {0.0, 0.0, travelling_wave::theory_mode}},
            {"field_phase", {0.0, 0.0, 0.0}},
            {"density_amplitude", {0.0, 0.0, travelling_wave::a_n_i}},
            {"density_phase", {0.0, 0.0, travelling_wave::phi_n_i}},
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
        {{"diagnostic", "FieldView"}, {"field", "ions/J"},
          {"out_dir", "ions/J"}},
        {{"diagnostic", "DistributionMoment"}, {"particles", "electrons"},
          {"moment", "density"}, {"out_dir", "electrons/density"}},
        {{"diagnostic", "DistributionMoment"}, {"particles", "electrons"},
          {"moment", "temperature_parallel"},
          {"out_dir", "electrons/temperature_parallel"}},
        {{"diagnostic", "DistributionMoment"}, {"particles", "ions"},
          {"moment", "density"}, {"out_dir", "ions/density"}},
        {{"diagnostic", "DistributionMoment"}, {"particles", "ions"},
          {"moment", "temperature_parallel"},
          {"out_dir", "ions/temperature_parallel"}},
        {
          {"diagnostic", "DkDistributionFunction"},
          {"particles", "electrons"},
          {"out_dir", "electrons/distribution_function"},
          {"v_parallel", {{"min", -travelling_wave::vpar_max_e},
            {"max", travelling_wave::vpar_max_e}, {"bins", 3000}}},
          {"mu_p", {{"min", 0.0}, {"max", travelling_wave::mu_max_e},
            {"bins", 1}}},
          {"diagnose_period", travelling_wave::diagnostic_period},
          {"max_frames", 5},
        },
        {
          {"diagnostic", "DkDistributionFunction"},
          {"particles", "ions"},
          {"out_dir", "ions/distribution_function"},
          {"v_parallel", {{"min", -travelling_wave::vpar_max_i},
            {"max", travelling_wave::vpar_max_i}, {"bins", 192}}},
          {"mu_p", {{"min", 0.0}, {"max", travelling_wave::mu_max_i},
            {"bins", 1}}},
          {"diagnose_period", travelling_wave::diagnostic_period},
          {"max_frames", 5},
        },
      },
    },
  });
}
