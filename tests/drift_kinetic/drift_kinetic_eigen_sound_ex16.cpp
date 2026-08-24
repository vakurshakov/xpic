#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Ion-sound Landau-quasimode test for the drift_kinetic implementation.    \n"
  "The density, parallel flux, and second moment of each species are         \n"
  "initialized from the full kinetic dispersion root.  A positive local     \n"
  "Gaussian realizes those three moments.  It is not a literal damped       \n"
  "eigenfunction (none exists as a regular real-velocity PDF), but it        \n"
  "suppresses the fast/nonmodal transient and leaves the forward Landau      \n"
  "pole dominant.                                                           \n";

namespace eigen {

// Inputs consumed by test_ion_sound_constants.py.  Keep them as literal
// constexpr values so changing a physical/grid parameter makes the fast
// theory-consistency CTest request regenerated mode constants.
constexpr double theory_mec2_kev = 511.0;
constexpr double theory_Te_kev = 10.0;
constexpr double theory_Ti_kev = 0.1;
constexpr double theory_ne = 1.0;
constexpr double theory_ni = 1.0;
constexpr double theory_qe = -1.0;
constexpr double theory_qi = 1.0;
constexpr double theory_me = 1.0;
constexpr double theory_mi = 100.0;
constexpr double theory_Lz = 75.0;
constexpr double theory_dz = 2.5;
constexpr double theory_mode = 1.0;
constexpr double theory_dn_i = 0.03;

// The full-Z dispersion relation includes the discrete electrostatic coupling
// S_E*S_rho*k/k_h = sinc(k*dz/2)^4.  E_force is the harmonic after S1
// interpolation to particles; E_grid=E_force/sinc^2 is stored on Yee faces.
constexpr double E_force = 4.8901683564625230e-05;
constexpr double E_grid = 4.9080831959185819e-05;
constexpr double omega_real = 1.1840593675280411e-03;
constexpr double gamma = 7.2904009887983382e-05;

constexpr double a_n_e = 2.9995906312399831e-02;
constexpr double phi_n_e = -3.0147264125957078;
constexpr double a_n_i = 3.0000000000000002e-02;
constexpr double phi_n_i = -3.0147438177476489;
// SetCosineField stores E_z at z-index g while the Yee E_z location is
// (g+1/2) dz.  Advancing the stored harmonic by k dz/2 makes the field seen by
// particles a zero-phase cosine, matching field_phase below.
constexpr double field_grid_phase = 1.0471975511965977e-01;

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
  dx = eigen::theory_dz;
  geom_nx = 3;
  geom_x = geom_nx * dx;
  geom_ny = 3;
  geom_y = geom_ny * dx;
  geom_nz = static_cast<PetscInt>(eigen::theory_Lz / dx);
  geom_z = geom_nz * dx;

  dt = 5;
  // ceil(5 * (2*pi/omega_real) / dt): retain five periods for a stable
  // damping fit without evolving the mode deep into the marker-noise floor.
  geom_nt = 5307;
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
        {"Np", 2048},
        {"n", eigen::theory_ne},
        {"q", eigen::theory_qe},
        {"m", eigen::theory_me},
        {"Tx", 0.0},
        {"Ty", 0.0},
        {"Tz", eigen::theory_Te_kev},
        {"coord_is_gc", true},
      },
      {
        {"sort_name", "ions"},
        {"Np", 2048},
        {"n", eigen::theory_ni},
        {"q", eigen::theory_qi},
        {"m", eigen::theory_mi},
        {"T", eigen::theory_Ti_kev},
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
              {"amplitude", {0.0, 0.0, eigen::E_grid}},
              {"wave_number", {0.0, 0.0, eigen::theory_mode}},
              {"phase", {0.0, 0.0, eigen::field_grid_phase}},
            },
          },
        },
        {
          {"command", "SetParticles"},
          {"particles", "electrons"},
          {"coordinate", {
            {"name", "CoordinateInBoxQuietSineExactLatticePaired"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"amplitude", {0.0, 0.0, eigen::a_n_e}},
            {"wave_number", {0.0, 0.0, eigen::theory_mode}},
            {"phase", {0.0, 0.0, eigen::phi_n_e}},
          }},
          {"momentum", {
            {"name", "KineticIonSoundMomentsQuiet"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"force_electric_amplitude", eigen::E_force},
            {"omega_real", eigen::omega_real},
            {"gamma", eigen::gamma},
            {"wave_number", {0.0, 0.0, eigen::theory_mode}},
            {"field_phase", {0.0, 0.0, 0.0}},
            {"density_amplitude", {0.0, 0.0, eigen::a_n_e}},
            {"density_phase", {0.0, 0.0, eigen::phi_n_e}},
          }},
        },
        {
          {"command", "SetParticles"},
          {"particles", "ions"},
          {"coordinate", {
            {"name", "CoordinateInBoxQuietSineExactLatticePaired"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"amplitude", {0.0, 0.0, eigen::a_n_i}},
            {"wave_number", {0.0, 0.0, eigen::theory_mode}},
            {"phase", {0.0, 0.0, eigen::phi_n_i}},
          }},
          {"momentum", {
            {"name", "KineticIonSoundMomentsQuiet"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"force_electric_amplitude", eigen::E_force},
            {"omega_real", eigen::omega_real},
            {"gamma", eigen::gamma},
            {"wave_number", {0.0, 0.0, eigen::theory_mode}},
            {"field_phase", {0.0, 0.0, 0.0}},
            {"density_amplitude", {0.0, 0.0, eigen::a_n_i}},
            {"density_phase", {0.0, 0.0, eigen::phi_n_i}},
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
