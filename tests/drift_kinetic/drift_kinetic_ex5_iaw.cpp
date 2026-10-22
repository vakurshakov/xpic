#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Drift-kinetic ion-acoustic-wave ringdown with mi/me = 25.                 \n"
  "                                                                          \n"
  "The periodic domain has Lz = 32, dz = 1, Te = 5 keV, Ti = 0.1 keV and     \n"
  "8192 markers per cell for each species.  The mode-1 perturbation has      \n"
  "dn/n = 0.03.  The mass ratio gives omega = 3.9304e-3 and Landau damping   \n"
  "Gamma = 4.7291e-4, so three wave periods fit in 960 steps with dt = 5.    \n"
  "                                                                          \n"
  "SetCosineField initializes Ez.  SetParticlesIonAcousticWave initializes   \n"
  "the matching density, parallel-flow and pressure moments of electrons and \n"
  "ions for the kinetic root, avoiding a counter-propagating branch caused   \n"
  "by a density-only perturbation.  Tx = Ty = 0, so the dynamics is one-     \n"
  "dimensional along the uniform magnetic field.                             \n"
  "                                                                          \n"
  "Density, parallel-temperature, current, E and B diagnostics describe the  \n"
  "ringdown.  Five early DistributionFunction frames retain the realized     \n"
  "f(z,v_parallel,mu) for linear-model and phase-space analysis with         \n"
  "drift_kinetic_ex5_iaw.py.                                                \n";

namespace ringdown {

// Linear-theory parameters used to generate the initialized mode.
constexpr double theory_mec2_kev = 511.0;
constexpr double theory_Te_kev = 5.0;
constexpr double theory_Ti_kev = 0.1;
constexpr double theory_ne = 1.0;
constexpr double theory_ni = 1.0;
constexpr double theory_qe = -1.0;
constexpr double theory_qi = 1.0;
constexpr double theory_me = 1.0;
constexpr double theory_mi = 25.0;
constexpr double theory_Lz = 32.0;
constexpr double theory_dz = 1.0;
constexpr double theory_mode = 1.0;
constexpr double theory_dn_i = 0.03;

// The full-Z dispersion relation includes the discrete electrostatic coupling
// S_E*S_rho*k/k_h = sinc(k*dz/2)^4.  E_force is the harmonic after S1
// interpolation to particles; E_grid=E_force/sinc^2 is stored on Yee faces.
constexpr double E_force = 5.6375367210036672e-05;
constexpr double E_grid = 5.6556837512382212e-05;
constexpr double omega_real = 3.9304083806753808e-03;
constexpr double gamma = 4.7291325876081436e-04;

constexpr double a_n_e = 2.9989220883451378e-02;
constexpr double phi_n_e = -2.8860882321865966e+00;
constexpr double a_n_i = 2.9999999999999995e-02;
constexpr double phi_n_i = -2.8861820860760221e+00;

// Velocity-space windows of the 5-D dumps: +-8 vT along the field, capped
// below the speed of light.  Tx = Ty = 0, so mu_p is identically zero for
// every marker and a single mu bin holds the whole distribution; the mu_max
// values below only record the thermal scale that a finite Tperp would need.
constexpr double vpar_max_e = 7.9134258824893067e-01;
constexpr double vpar_max_i = 2.2382548415701312e-02;
constexpr double mu_max_e = 2.3483365949119372e-01;
constexpr double mu_max_i = 2.3483365949119373e-03;

constexpr PetscInt particles_per_cell = 8192;
constexpr PetscInt cells_z = static_cast<PetscInt>(theory_Lz / theory_dz);
constexpr PetscReal time_step = 5.0;
// T = 2*pi/omega_real = 1598.61, so 960 steps of dt = 5 cover three periods.
// The first ~0.7 T are noise-free at the 10 % level and carry ~0.55 e-folding
// of clean decay; the rest documents the marker-noise plateau and the energy
// budget once the wave has sunk into it.
constexpr PetscInt time_steps = 960;
constexpr PetscReal diagnostic_period = 2.0 * time_step;

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
  geom_nx = 4;
  geom_ny = 4;
  geom_nz = ringdown::cells_z;
  dx = ringdown::theory_dz;
  dy = ringdown::theory_dz;
  dz = ringdown::theory_dz;
  geom_x = geom_nx * dx;
  geom_y = geom_ny * dy;
  geom_z = geom_nz * dz;

  dt = ringdown::time_step;
  geom_nt = ringdown::time_steps;
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", "drift_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {"DriftKineticPush",
      {
        {"coord_is_gc", true},
        {"fail_on_terminal_nonconvergence", true},
      }},
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
      {
        {
          {"sort_name", "electrons"},
          {"Np", ringdown::particles_per_cell},
          {"n", ringdown::theory_ne},
          {"q", ringdown::theory_qe},
          {"m", ringdown::theory_me},
          {"Tx", 0.0},
          {"Ty", 0.0},
          {"Tz", ringdown::theory_Te_kev},
        },
        {
          {"sort_name", "ions"},
          {"Np", ringdown::particles_per_cell},
          {"n", ringdown::theory_ni},
          {"q", ringdown::theory_qi},
          {"m", ringdown::theory_mi},
          {"Tx", 0.0},
          {"Ty", 0.0},
          {"Tz", ringdown::theory_Ti_kev},
        },
      },
    },
    {
      "Presets",
      {
        {
          {"command", "SetField"},
          {"field", "B"},
          {"setter",
            {
              {"name", "SetUniformField"},
              {"value", {0.0, 0.0, 1.0}},
            }},
        },
        {
          {"command", "SetField"},
          {"field", "E"},
          {"setter",
            {
              {"name", "SetCosineField"},
              {"amplitude", {0.0, 0.0, ringdown::E_grid}},
              {"wave_number", {0.0, 0.0, ringdown::theory_mode}},
            }},
        },
        {
          {"command", "SetParticlesIonAcousticWave"},
          {"particles", "electrons"},
          {"wave_number", {0.0, 0.0, ringdown::theory_mode}},
          {"density_amplitude", {0.0, 0.0, ringdown::a_n_e}},
          {"density_phase", {0.0, 0.0, ringdown::phi_n_e}},
          {"force_electric_amplitude", ringdown::E_force},
          {"omega_real", ringdown::omega_real},
          {"gamma", ringdown::gamma},
        },
        {
          {"command", "SetParticlesIonAcousticWave"},
          {"particles", "ions"},
          {"wave_number", {0.0, 0.0, ringdown::theory_mode}},
          {"density_amplitude", {0.0, 0.0, ringdown::a_n_i}},
          {"density_phase", {0.0, 0.0, ringdown::phi_n_i}},
          {"force_electric_amplitude", ringdown::E_force},
          {"omega_real", ringdown::omega_real},
          {"gamma", ringdown::gamma},
        },
      },
    },
    {
      "Diagnostics",
      {
        // The analysis script needs all three components to extract the
        // longitudinal electric-field harmonic.
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
          {"out_dir", "electrons/density"},
        },
        {
          {"diagnostic", "DistributionMoment"},
          {"particles", "electrons"},
          {"moment", "temperature_parallel"},
          {"out_dir", "electrons/temperature_parallel"},
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
          {"moment", "temperature_parallel"},
          {"out_dir", "ions/temperature_parallel"},
        },

        // Five early 5-D frames: enough for drift_kinetic_ex5_iaw.py to
        // start the exact linear evolution from the realized
        // distribution instead of the configured moments.  One mu bin,
        // see mu_max_* above.
        {
          {"diagnostic", "DistributionFunction"},
          {"particles", "electrons"},
          {"out_dir", "electrons/distribution_function"},
          {"v_parallel",
            {
              {"min", -ringdown::vpar_max_e},
              {"max", ringdown::vpar_max_e},
              {"bins", 3000},
            }},
          {"mu_p",
            {
              {"min", 0.0},
              {"max", ringdown::mu_max_e},
              {"bins", 1},
            }},
          {"diagnose_period", ringdown::diagnostic_period},
          {"max_frames", 5},
        },
        {
          {"diagnostic", "DistributionFunction"},
          {"particles", "ions"},
          {"out_dir", "ions/distribution_function"},
          {"v_parallel",
            {
              {"min", -ringdown::vpar_max_i},
              {"max", ringdown::vpar_max_i},
              {"bins", 192},
            }},
          {"mu_p",
            {
              {"min", 0.0},
              {"max", ringdown::mu_max_i},
              {"bins", 1},
            }},
          {"diagnose_period", ringdown::diagnostic_period},
          {"max_frames", 5},
        },
      },
    },
  });
}
