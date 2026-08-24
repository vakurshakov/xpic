#include "src/utils/particles_load.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
  if (!condition)
    throw std::runtime_error(message);
}

void test_antithetic_temperature()
{
  SortParameters params{};
  params.m = 1.0;
  params.Tx = 20.0;
  params.Ty = 10.0;
  params.Tz = 0.1;

  MaxwellianVelocityQuiet loader(params);
  constexpr std::size_t number_of_pairs = 65'536;
  PetscReal second_moment[3]{};

  for (std::size_t pair = 0; pair < number_of_pairs; ++pair) {
    const Vector3R first = loader(Vector3R{});
    const Vector3R second = loader(Vector3R{});

    require((first + second).abs_max() == 0.0,
      "MaxwellianVelocityQuiet pair is not exactly antithetic");
    require(first.squared() < 1.0 && second.squared() < 1.0,
      "MaxwellianVelocityQuiet emitted a superluminal velocity");

    for (Axis axis : {X, Y, Z})
      second_moment[axis] += first[axis] * first[axis];
  }

  const PetscReal temperatures[3] = {params.Tx, params.Ty, params.Tz};
  for (Axis axis : {X, Y, Z}) {
    const PetscReal measured =
      second_moment[axis] / static_cast<PetscReal>(number_of_pairs);
    const PetscReal expected = temperatures[axis] / (params.m * mec2);
    const PetscReal relative_error = std::abs(measured / expected - 1.0);
    require(relative_error < 0.005,
      "MaxwellianVelocityQuiet temperature error exceeds 0.5%");
  }
}

void test_subluminal_rejection()
{
  SortParameters params{};
  params.m = 1.0;
  params.Tx = mec2;
  params.Ty = mec2;
  params.Tz = mec2;

  MaxwellianVelocityQuiet loader(params);
  for (std::size_t pair = 0; pair < 128; ++pair) {
    const Vector3R first = loader(Vector3R{});
    const Vector3R second = loader(Vector3R{});
    require(first.squared() < 1.0 && second.squared() < 1.0,
      "MaxwellianVelocityQuiet rejection did not enforce |v| < 1");
    require((first + second).abs_max() == 0.0,
      "MaxwellianVelocityQuiet rejection broke antithetic pairing");
  }
}

void test_exact_sine_lattice_harmonic()
{
  const BoxGeometry box{
    Vector3R{4.0, -7.0, 50.0}, Vector3R{34.0, 23.0, 250.0}};
  const Vector3R amplitude{0.0, 0.0, 0.03};
  const Vector3R wave_number{0.0, 0.0, 1.0};
  const Vector3R phase{0.0, 0.0, -3.015684};
  constexpr std::size_t number_of_pairs = 65'536;
  CoordinateInBoxQuietSineExactPaired loader(
    box, amplitude, wave_number, phase, number_of_pairs);

  std::complex<PetscReal> measured_harmonic{};
  for (std::size_t pair = 0; pair < number_of_pairs; ++pair) {
    const Vector3R first = loader();
    const Vector3R second = loader();
    require((first - second).abs_max() == 0.0,
      "exact-sine lattice coordinate is not paired");
    const PetscReal theta =
      2.0 * M_PI * (first[Z] - box.min[Z]) /
      (box.max[Z] - box.min[Z]);
    measured_harmonic += std::polar<PetscReal>(1.0, -theta);
  }
  measured_harmonic *=
    2.0 / static_cast<PetscReal>(number_of_pairs);

  const std::complex<PetscReal> imaginary{0.0, 1.0};
  const std::complex<PetscReal> expected_harmonic =
    -imaginary * amplitude[Z] * std::exp(imaginary * phase[Z]);
  require(std::abs(measured_harmonic - expected_harmonic) < 1.0e-10,
    "exact-sine lattice first harmonic is inaccurate");

  bool exhausted_threw = false;
  try {
    static_cast<void>(loader());
  }
  catch (const std::runtime_error&) {
    exhausted_threw = true;
  }
  require(exhausted_threw,
    "exact-sine lattice accepted more pairs than configured");
}

void test_ion_sound_moment_pair()
{
  SortParameters params{};
  params.n = 1.0;
  params.q = 1.0;
  params.m = 100.0;
  params.Tx = 0.1;
  params.Ty = 0.1;
  params.Tz = 0.1;

  const BoxGeometry box{
    Vector3R{4.0, -7.0, 50.0}, Vector3R{34.0, 23.0, 250.0}};
  constexpr PetscReal electric_amplitude = 3.667853669e-5;
  constexpr PetscReal omega_real = 6.232226583e-4;
  constexpr PetscReal gamma = 3.869058784e-5;
  const Vector3R wave_number{0.0, 0.0, 1.0};
  const Vector3R field_phase{0.0, 0.0, 0.17};
  const Vector3R density_amplitude{0.0, 0.0, 0.03};
  const Vector3R density_phase{0.0, 0.0, -3.015684 + 0.17};
  KineticIonSoundMomentsQuiet loader(params, box, electric_amplitude,
    omega_real, gamma, wave_number, field_phase,
    density_amplitude, density_phase);

  const Vector3R coordinate{13.0, 2.0, 87.0};
  const PetscReal k = 2.0 * M_PI / 200.0;
  const PetscReal theta = k * (coordinate[Z] - box.min[Z]);
  const std::complex<PetscReal> imaginary{0.0, 1.0};
  const std::complex<PetscReal> omega{omega_real, -gamma};
  const std::complex<PetscReal> density_harmonic =
    -imaginary * params.n * density_amplitude[Z] *
    std::exp(imaginary * density_phase[Z]);
  const std::complex<PetscReal> flux_harmonic =
    omega * density_harmonic / k;
  const std::complex<PetscReal> field_harmonic =
    electric_amplitude * std::exp(imaginary * field_phase[Z]);
  const std::complex<PetscReal> second_harmonic =
    (omega * flux_harmonic -
      imaginary * (params.q / params.m) * params.n * field_harmonic) / k;
  const std::complex<PetscReal> spatial_phase =
    std::exp(imaginary * theta);
  const PetscReal density =
    params.n + std::real(density_harmonic * spatial_phase);
  const PetscReal flux = std::real(flux_harmonic * spatial_phase);
  const PetscReal equilibrium_variance = params.Tz / (params.m * mec2);
  const PetscReal second_moment = params.n * equilibrium_variance +
    std::real(second_harmonic * spatial_phase);
  const PetscReal expected_bulk = flux / density;
  const PetscReal expected_variance =
    second_moment / density - expected_bulk * expected_bulk;

  constexpr std::size_t number_of_pairs = 65'536;
  PetscReal measured_second_moment[3]{};
  for (std::size_t pair = 0; pair < number_of_pairs; ++pair) {
    const Vector3R first = loader(coordinate);
    const Vector3R second = loader(coordinate);
    const Vector3R measured_bulk = 0.5 * (first + second);
    require(std::abs(measured_bulk[Z] - expected_bulk) < 1.0e-15 &&
        measured_bulk[X] == 0.0 && measured_bulk[Y] == 0.0,
      "ion-sound moment pair does not have the prescribed local flux");
    require(first.squared() < 1.0 && second.squared() < 1.0,
      "ion-sound moment loader emitted a superluminal velocity");

    for (Axis axis : {X, Y, Z})
      measured_second_moment[axis] +=
        0.5 * (first[axis] * first[axis] + second[axis] * second[axis]);
  }

  const PetscReal expected_second_moment[3] = {
    params.Tx / (params.m * mec2),
    params.Ty / (params.m * mec2),
    expected_variance + expected_bulk * expected_bulk,
  };
  for (Axis axis : {X, Y, Z}) {
    const PetscReal measured =
      measured_second_moment[axis] /
      static_cast<PetscReal>(number_of_pairs);
    require(std::abs(measured / expected_second_moment[axis] - 1.0) < 0.005,
      "ion-sound moment loader second-moment error exceeds 0.5%");
  }

  KineticIonSoundMomentsQuiet mismatched_pair_loader(params, box,
    electric_amplitude, omega_real, gamma, wave_number, field_phase,
    density_amplitude, density_phase);
  static_cast<void>(mismatched_pair_loader(coordinate));
  bool mismatch_threw = false;
  try {
    static_cast<void>(mismatched_pair_loader(
      coordinate + Vector3R{0.0, 0.0, 1.0}));
  }
  catch (const std::runtime_error&) {
    mismatch_threw = true;
  }
  require(mismatch_threw,
    "ion-sound moment loader accepted a mismatched quiet pair");
}

void test_ion_sound_field_pressure_sign()
{
  SortParameters params{};
  params.n = 2.5;
  params.q = -2.0;
  params.m = 4.0;
  params.Tx = 1.0;
  params.Ty = 1.0;
  params.Tz = 1.0;

  const BoxGeometry box{Vector3R{}, Vector3R{30.0, 30.0, 200.0}};
  constexpr PetscReal electric_amplitude = 2.0e-6;
  constexpr PetscReal omega_real = 6.0e-4;
  constexpr PetscReal gamma = 4.0e-5;
  const Vector3R wave_number{0.0, 0.0, 1.0};
  const Vector3R zero{};
  KineticIonSoundMomentsQuiet loader(params, box, electric_amplitude,
    omega_real, gamma, wave_number, zero, zero, zero);

  // At kz=pi/2 and M0=M1=0, the independently integrated force moment is
  // M2/n0 = vT^2 + (q/m) E0/k.  A negative charge must reduce the variance.
  const Vector3R coordinate{15.0, 15.0, 50.0};
  const PetscReal k = 2.0 * M_PI / 200.0;
  const PetscReal expected_variance = params.Tz / (params.m * mec2) +
    (params.q / params.m) * electric_amplitude / k;

  constexpr std::size_t number_of_pairs = 65'536;
  PetscReal measured_second_moment = 0.0;
  for (std::size_t pair = 0; pair < number_of_pairs; ++pair) {
    const Vector3R first = loader(coordinate);
    const Vector3R second = loader(coordinate);
    require((first + second).abs_max() == 0.0,
      "field-only ion-sound pair has non-zero flux");
    measured_second_moment +=
      0.5 * (first[Z] * first[Z] + second[Z] * second[Z]);
  }
  measured_second_moment /= static_cast<PetscReal>(number_of_pairs);
  require(std::abs(measured_second_moment / expected_variance - 1.0) < 0.005,
    "field-only ion-sound pressure has the wrong sign or normalization");
}

void test_ion_sound_invalid_moments_report()
{
  SortParameters params{};
  params.sort_name = "ions";
  params.n = 1.0;
  params.q = 1.0;
  params.m = 100.0;
  params.Tx = 0.1;
  params.Ty = 0.1;
  params.Tz = 0.1;

  const BoxGeometry box{Vector3R{}, Vector3R{7.5, 7.5, 75.0}};
  const Vector3R wave_number{0.0, 0.0, 1.0};
  const Vector3R zero{};
  const Vector3R density_amplitude{0.0, 0.0, 0.03};
  const Vector3R stale_density_phase{0.0, 0.0, -3.0156843733207972};

  bool stale_threw = false;
  std::string stale_message;
  try {
    KineticIonSoundMomentsQuiet stale(params, box,
      3.6678438440940783e-05, 6.2322182890254662e-04,
      3.8690499431800553e-05, wave_number, zero,
      density_amplitude, stale_density_phase);
  }
  catch (const std::runtime_error& error) {
    stale_threw = true;
    stale_message = error.what();
  }
  require(stale_threw,
    "ion-sound loader accepted a mode with negative local variance");
  require(stale_message.find("ions") != std::string::npos &&
      stale_message.find("P_min=-") != std::string::npos &&
      stale_message.find("ion_sound.py --theory --grid-dz") !=
        std::string::npos,
    "ion-sound positivity error does not identify the failing margin and remedy");

  const Vector3R corrected_density_phase{
    0.0, 0.0, -3.0147438177476489};
  KineticIonSoundMomentsQuiet corrected(params, box,
    4.8901683564625230e-05, 1.1840593675280411e-03,
    7.2904009887983382e-05, wave_number, zero,
    density_amplitude, corrected_density_phase);
  const Vector3R coordinate{3.75, 3.75, 37.5};
  const Vector3R first = corrected(coordinate);
  const Vector3R second = corrected(coordinate);
  require(first.squared() < 1.0 && second.squared() < 1.0,
    "corrected ex15 ion-sound mode emitted a superluminal pair");
}

}  // namespace

int main()
{
  try {
    test_antithetic_temperature();
    test_subluminal_rejection();
    test_exact_sine_lattice_harmonic();
    test_ion_sound_moment_pair();
    test_ion_sound_field_pressure_sign();
    test_ion_sound_invalid_moments_report();
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
