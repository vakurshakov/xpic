#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Ion-sound ringdown at mi/me = 25: a fast-damping variant of ringdown_ex1  \n"
  "designed so that the measured Landau rate is limited by marker noise      \n"
  "alone, not by the initial condition.                                      \n"
  "                                                                          \n"
  "Three changes with respect to ringdown_ex1 (Lz = 32, dz = 1, Te = 5 keV,  \n"
  "Ti = 0.1 keV, mode 1, dn/n = 0.03 are all kept):                          \n"
  "                                                                          \n"
  "  1. mi/me = 100 -> 25.  Gamma/omega ~ sqrt(me/mi) grows 0.060 -> 0.120   \n"
  "     and omega ~ 1/sqrt(mi) shortens the period 3153 -> 1599, so the wave \n"
  "     reaches the noise floor in 4x fewer steps.  Crucially zeta_i =       \n"
  "     omega/(sqrt(2) k vTi) ~ sqrt(Te/2Ti) does NOT depend on mi, so it    \n"
  "     stays at 5.06 and ion Landau damping remains negligible.  Raising Ti \n"
  "     to 1 keV would reach the same Gamma/omega but drop zeta_i to 2.15,   \n"
  "     where the Landau pole stops dominating: the exact linear response    \n"
  "     then deviates from a pure exponential by ~29 % (vs ~7 % here) and a  \n"
  "     naive exponential fit overestimates Gamma by ~8 % in theory alone.   \n"
  "                                                                          \n"
  "  2. Eigenmode loading, as in eigen_sound_ex15: SetCosineField for E plus \n"
  "     KineticIonSoundMomentsQuiet, which matches M0, M1 and M2 of the      \n"
  "     kinetic root.  ringdown_ex1 loads a density perturbation at zero     \n"
  "     bulk velocity, which excites both branches in equal parts            \n"
  "     (|A-|/|A+| ~ 1) and adds a ballistic transient; fitting that mixture \n"
  "     over half a period inflates Gamma by ~40 % even in exact theory.     \n"
  "     CoordinateInBoxQuietSineExactPaired removes the loading noise \n"
  "     of the initial state (~5e-8 instead of ~9e-5 in the m = 2..9 rms).   \n"
  "                                                                          \n"
  "  3. Np = 1024 -> 8192 per cell.  The equilibrium marker-noise plateau of \n"
  "     the relative density is ~0.48 sqrt(2/N_tot) per mode and falls only  \n"
  "     as 1/sqrt(N_tot); the observable dynamic range is ln(a0/noise), so   \n"
  "     8x more markers buys 1.81 -> 2.85 e-foldings of envelope decay.      \n"
  "     Note that the quiet start only delays the plateau by ~0.1 T -- the   \n"
  "     marker count, not the loader, sets the floor after that.             \n"
  "                                                                          \n"
  "The diagnostics follow eigen_sound_ex15 rather than the ringdown family:  \n"
  "the three-component E dump, the parallel-temperature moment and the       \n"
  "DkDistributionFunction frames are what ion_sound.py needs to start the    \n"
  "exact linear theory from the realized f and to separate wave energy from  \n"
  "irreversible heating.  Ez itself stays below the noise floor at this      \n"
  "marker count and cannot be de-aliased by dumping more often: omega_pe*dt  \n"
  "= 5 means the step is already longer than the plasma period.              \n";

namespace ringdown {

// Inputs consumed by test_ion_sound_constants.py.  Keep them as literal
// constexpr values so changing a physical/grid parameter makes the fast
// theory-consistency CTest request regenerated mode constants.
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
// SetCosineField stores E_z at z-index g while the Yee E_z location is
// (g+1/2) dz.  Advancing the stored harmonic by k dz/2 makes the field seen by
// particles a zero-phase cosine, matching field_phase below.
constexpr double field_grid_phase = 9.8174770424681035e-02;

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
        {"diagnose_period", ringdown::diagnostic_period},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "electrons"},
        {"Np", ringdown::particles_per_cell},
        {"n", ringdown::theory_ne},
        {"q", ringdown::theory_qe},
        {"m", ringdown::theory_me},
        {"Tx", 0.0},
        {"Ty", 0.0},
        {"Tz", ringdown::theory_Te_kev},
        {"coord_is_gc", true},
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
              {"amplitude", {0.0, 0.0, ringdown::E_grid}},
              {"wave_number", {0.0, 0.0, ringdown::theory_mode}},
              {"phase", {0.0, 0.0, ringdown::field_grid_phase}},
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
            {"amplitude", {0.0, 0.0, ringdown::a_n_e}},
            {"wave_number", {0.0, 0.0, ringdown::theory_mode}},
            {"phase", {0.0, 0.0, ringdown::phi_n_e}},
          }},
          {"momentum", {
            {"name", "KineticIonSoundMomentsQuiet"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"force_electric_amplitude", ringdown::E_force},
            {"omega_real", ringdown::omega_real},
            {"gamma", ringdown::gamma},
            {"wave_number", {0.0, 0.0, ringdown::theory_mode}},
            {"field_phase", {0.0, 0.0, 0.0}},
            {"density_amplitude", {0.0, 0.0, ringdown::a_n_e}},
            {"density_phase", {0.0, 0.0, ringdown::phi_n_e}},
          }},
        },
        {
          {"command", "SetParticles"},
          {"particles", "ions"},
          {"coordinate", {
            {"name", "CoordinateInBoxQuietSineExactPaired"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"amplitude", {0.0, 0.0, ringdown::a_n_i}},
            {"wave_number", {0.0, 0.0, ringdown::theory_mode}},
            {"phase", {0.0, 0.0, ringdown::phi_n_i}},
          }},
          {"momentum", {
            {"name", "KineticIonSoundMomentsQuiet"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"force_electric_amplitude", ringdown::E_force},
            {"omega_real", ringdown::omega_real},
            {"gamma", ringdown::gamma},
            {"wave_number", {0.0, 0.0, ringdown::theory_mode}},
            {"field_phase", {0.0, 0.0, 0.0}},
            {"density_amplitude", {0.0, 0.0, ringdown::a_n_i}},
            {"density_phase", {0.0, 0.0, ringdown::phi_n_i}},
          }},
        },
      },
    },
    {
      "Diagnostics",
      {
        // Full three-component E, unlike the {"component", "z"} dumps of the
        // older ringdown cases: ion_sound.electric_harmonic_series() expects
        // the vector layout.
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
        // Five early 5-D frames: enough for ion_sound.py --model to start the
        // exact linear evolution from the realized distribution instead of the
        // configured moments.  One mu bin, see mu_max_* above.
        {
          {"diagnostic", "DkDistributionFunction"},
          {"particles", "electrons"},
          {"out_dir", "electrons/distribution_function"},
          {"v_parallel", {
            {"min", -ringdown::vpar_max_e}, {"max", ringdown::vpar_max_e},
            {"bins", 3000}}},
          {"mu_p", {
            {"min", 0.0}, {"max", ringdown::mu_max_e}, {"bins", 1}}},
          {"diagnose_period", ringdown::diagnostic_period},
          {"max_frames", 5},
        },
        {
          {"diagnostic", "DkDistributionFunction"},
          {"particles", "ions"},
          {"out_dir", "ions/distribution_function"},
          {"v_parallel", {
            {"min", -ringdown::vpar_max_i}, {"max", ringdown::vpar_max_i},
            {"bins", 192}}},
          {"mu_p", {
            {"min", 0.0}, {"max", ringdown::mu_max_i}, {"bins", 1}}},
          {"diagnose_period", ringdown::diagnostic_period},
          {"max_frames", 5},
        },
      },
    },
  });
}
