#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Amplitude twin of ringdown_ex10: dn/n = 0.03 -> 0.01 at Np = 8192.       \n"
  "                                                                          \n"
  "Everything else -- Lz = 32, dz = 1, dt = 5, mode 1, mi/me = 25, Te = 5    \n"
  "keV, Ti = 0.1 keV, the eigenmode loading, the diagnostics and the 960     \n"
  "steps -- is byte-for-byte ex10, so the pair isolates the wave amplitude   \n"
  "as the single variable.                                                   \n"
  "                                                                          \n"
  "What the pair measures.  In the ex10..ex13 marker scan the broadband      \n"
  "floor of the density harmonics splits into two parts, nu^2 = P^2 + Q/N:   \n"
  "                                                                          \n"
  "  * Q/N is ordinary marker shot noise.  It dominates up to t ~ 0.4 T,     \n"
  "    where the measured exponent of nu ~ N^-p is 0.46..0.53.               \n"
  "  * P does not depend on N at all.  Past t ~ 0.6 T it dominates and the   \n"
  "    exponent collapses to 0.12..0.27, which is why 8x more markers buy    \n"
  "    only 0.20 T of extra clean window per doubling instead of the         \n"
  "    ln2/(2 Gamma) = 0.46 T that a pure marker floor would give.           \n"
  "                                                                          \n"
  "The N-independent part is almost certainly nonlinear.  The second         \n"
  "harmonic is identical to three digits across the whole ex10..ex13 scan    \n"
  "(|dn_2| = 5.90, 6.23, 6.26, 6.20 e-4 at t = 0.2 T for Np = 1024..8192)    \n"
  "while the marker floor changes by 2.8x, and omega(2k) = 2 omega(k) for    \n"
  "ion sound puts the quadratic drive exactly on the m = 2 resonance.  At    \n"
  "this amplitude the wave is not linear for the electrons either:           \n"
  "omega_b/omega = 0.85 and omega_b/Gamma = 7.0.                             \n"
  "                                                                          \n"
  "Prediction under test.  A nonlinear P scales as dn^2 while the signal     \n"
  "scales as dn, so 0.03 -> 0.01 must drop P by 9x, |dn_2| by 9x, and the    \n"
  "relative floor P/|dn_1| by 3x.  The marker part Q/N must not move at all: \n"
  "it is set by the marker count, not by the wave.  If instead P falls by    \n"
  "3x or stays put, the floor is not the wave nonlinearity and the ex10      \n"
  "diagnosis has to be redone.                                               \n"
  "                                                                          \n"
  "Cost of the lower amplitude: the early, marker-limited phase gets 3x      \n"
  "worse in relative terms, so this run alone is a WORSE ringdown            \n"
  "measurement than ex10.  It is a diagnostic of the floor, not a better     \n"
  "damping measurement.                                                      \n";

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
constexpr double theory_dn_i = 0.01;

// The full-Z dispersion relation includes the discrete electrostatic coupling
// S_E*S_rho*k/k_h = sinc(k*dz/2)^4.  E_force is the harmonic after S1
// interpolation to particles; E_grid=E_force/sinc^2 is stored on Yee faces.
constexpr double E_force = 1.8791789070012225e-05;
constexpr double E_grid = 1.8852279170794072e-05;
constexpr double omega_real = 3.9304083806753808e-03;
constexpr double gamma = 4.7291325876081436e-04;

constexpr double a_n_e = 9.9964069611504624e-03;
constexpr double phi_n_e = -2.8860882321865966e+00;
constexpr double a_n_i = 1.0000000000000000e-02;
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
// Same length as ex10 on purpose: the two runs are compared frame by frame.
// At a third of the ex10 amplitude the marker floor is reached ~1.1 e-folding
// of signal earlier, so expect the clean window to SHRINK; what must shrink
// faster is |dn_2| and the N-independent part of the m >= 3 floor.
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
            {"name", "CoordinateInBoxQuietSineExactLatticePaired"},
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
            {"name", "CoordinateInBoxQuietSineExactLatticePaired"},
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
