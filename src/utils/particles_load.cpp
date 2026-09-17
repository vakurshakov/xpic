#include "particles_load.h"

#include "src/utils/halton_generator.h"
#include "src/utils/random_generator.h"
#include "src/utils/utils.h"

#include <limits>
#include <stdexcept>

Vector3R PreciseCoordinate::operator()()
{
  return dot;
}

Vector3R CoordinateInBox::operator()()
{
  return Vector3R{
    box.min[X] + random_01() * (box.max[X] - box.min[X]),
    box.min[Y] + random_01() * (box.max[Y] - box.min[Y]),
    box.min[Z] + random_01() * (box.max[Z] - box.min[Z]),
  };
}

Vector3R CoordinateInCylinder::operator()()
{
  PetscReal r = cyl.radius * std::sqrt(random_01());
  PetscReal phi = 2.0 * M_PI * random_01();

  return Vector3R{
    cyl.center[X] + r * std::cos(phi),
    cyl.center[Y] + r * std::sin(phi),
    cyl.center[Z] + cyl.height * (random_01() - 0.5),
  };
}

static PetscReal sine_density_inv_cdf(
  PetscReal u, PetscReal amplitude, PetscReal mode, PetscReal phase)
{
  PetscReal x_lo = 0.0;
  PetscReal x_hi = 1.0;
  PetscReal x = u;
  const PetscReal inv_k = 1.0 / (2.0 * M_PI * mode);

  for (int it = 0; it < 64; ++it) {
    const PetscReal arg = 2.0 * M_PI * mode * x + phase;
    const PetscReal cdf =
      x + amplitude * inv_k * (std::cos(phase) - std::cos(arg));
    const PetscReal pdf = 1.0 + amplitude * std::sin(arg);

    if (cdf < u)
      x_lo = x;
    else
      x_hi = x;

    PetscReal x_new = x - (cdf - u) / pdf;
    if (x_new <= x_lo || x_new >= x_hi)
      x_new = 0.5 * (x_lo + x_hi);
    if (std::abs(x_new - x) < 16.0 * std::numeric_limits<PetscReal>::epsilon())
      return x_new;
    x = x_new;
  }
  return x;
}

CoordinateIonSoundPaired::CoordinateIonSoundPaired(
  BoxGeometry box, const Vector3R& amplitude, const Vector3R& wave_number,
  const Vector3R& phase, std::size_t active_axis_lattice_pairs)
  : box(std::move(box)), amplitude(amplitude), wave_number(wave_number),
    phase(phase), active_axis_lattice_pairs(active_axis_lattice_pairs)
{
  PetscInt active = 0;
  for (Axis a : {X, Y, Z}) {
    if (amplitude[a] == 0.0 && wave_number[a] == 0.0)
      continue;
    if (amplitude[a] == 0.0 || wave_number[a] == 0.0)
      throw std::runtime_error(
        "CoordinateIonSoundPaired requires amplitude and wave_number together");
    if (std::abs(amplitude[a]) >= 1.0)
      throw std::runtime_error(
        "CoordinateIonSoundPaired requires |amplitude| < 1");
    const PetscReal rounded_mode = std::round(wave_number[a]);
    if (rounded_mode <= 0.0 ||
        std::abs(wave_number[a] - rounded_mode) > 1.0e-12)
      throw std::runtime_error(
        "CoordinateIonSoundPaired requires a positive integer wave_number");
    axis = a;
    ++active;
  }
  if (active != 1)
    throw std::runtime_error(
      "CoordinateIonSoundPaired requires exactly one active axis");
}

Vector3R CoordinateIonSoundPaired::operator()()
{
  if (return_second) {
    return_second = false;
    return paired_coordinate;
  }

  const std::size_t index = pair_counter + 1;
  for (Axis a : {X, Y, Z}) {
    PetscReal u;
    if (a == axis && active_axis_lattice_pairs != 0) {
      if (pair_counter >= active_axis_lattice_pairs)
        throw std::runtime_error(
          "CoordinateIonSoundPaired exhausted its active-axis lattice");
      u = (static_cast<PetscReal>(pair_counter) + 0.5) /
        static_cast<PetscReal>(active_axis_lattice_pairs);
    }
    else {
      u = halton::van_der_corput(
        index, halton::coordinate_bases[static_cast<std::size_t>(a)]);
    }
    if (a == axis) {
      u = sine_density_inv_cdf(
        u, amplitude[a], wave_number[a], phase[a]);
    }
    paired_coordinate[a] =
      box.min[a] + u * (box.max[a] - box.min[a]);
  }

  ++pair_counter;
  return_second = true;
  return paired_coordinate;
}

Vector3R CoordinateInBoxCosineHump::operator()()
{
  const PetscReal center = 0.5 * (box.min[axis] + box.max[axis]);
  const PetscReal half_length = 0.5 * (box.max[axis] - box.min[axis]);

  while (true) {
    Vector3R coordinate{
      box.min[X] + random_01() * (box.max[X] - box.min[X]),
      box.min[Y] + random_01() * (box.max[Y] - box.min[Y]),
      box.min[Z] + random_01() * (box.max[Z] - box.min[Z]),
    };
    const PetscReal normalized = (coordinate[axis] - center) / half_length;
    const PetscReal weight = 0.5 * (1.0 + std::cos(M_PI * normalized));
    if (random_01() < weight)
      return coordinate;
  }
}

Vector3R CoordinateOnAnnulus::operator()()
{
  PetscReal r = std::sqrt(POW2(ann.inner_r) +  //
    (POW2(ann.outer_r) - POW2(ann.inner_r)) * random_01());

  PetscReal phi = 2.0 * M_PI * random_01();

  return Vector3R{
    ann.center[X] + r * std::cos(phi),
    ann.center[Y] + r * std::sin(phi),
    ann.center[Z] + ann.height * (random_01() - 0.5),
  };
}

Vector3R PreciseMomentum::operator()(const Vector3R& /* coordinate */)
{
  return value;
}

PetscReal temperature_momentum(PetscReal temperature, PetscReal mass)
{
  return std::sqrt(-2.0 * (temperature * mass / mec2) * std::log(random_01()));
}

Vector3R MaxwellianMomentum::operator()(const Vector3R& /* coordinate */)
{
  Vector3R result{
    params.px +
      std::sin(2.0 * M_PI * random_01()) *
        temperature_momentum(params.Tx, params.m),

    params.py +
      std::sin(2.0 * M_PI * random_01()) *
        temperature_momentum(params.Ty, params.m),

    params.pz +
      std::sin(2.0 * M_PI * random_01()) *
        temperature_momentum(params.Tz, params.m),
  };

  if (tov)
    result /= std::sqrt(params.m * params.m + result.squared());
  return result;
}

static Vector3R sample_thermal_velocity(const SortParameters& params)
{
  Vector3R result{
    std::sin(2.0 * M_PI * random_01()) * temperature_momentum(params.Tx, params.m),
    std::sin(2.0 * M_PI * random_01()) * temperature_momentum(params.Ty, params.m),
    std::sin(2.0 * M_PI * random_01()) * temperature_momentum(params.Tz, params.m),
  };
  result /= std::sqrt(params.m * params.m + result.squared());
  return result;
}

static Vector3R sample_maxwellian_velocity_quiet(
  const SortParameters& params, std::size_t index)
{
  const PetscReal temperatures[3] = {params.Tx, params.Ty, params.Tz};

  Vector3R result;
  for (Axis a : {X, Y, Z}) {
    const auto slot = static_cast<std::size_t>(a);
    const PetscReal sigma =
      std::sqrt(temperatures[slot] / (params.m * mec2));
    const PetscReal radius = std::sqrt(-2.0 * std::log(halton::van_der_corput(
      index, halton::velocity_radius_bases[slot])));
    const PetscReal phase =
      2.0 * M_PI * halton::van_der_corput(index, halton::velocity_phase_bases[slot]);
    result[a] = sigma * radius * std::sin(phase);
  }
  return result;
}

static Vector3R sine_velocity_shift(const BoxGeometry& box, const Vector3R& velocity,
  const Vector3R& wave_number, const Vector3R& phase,
  const Vector3R& coordinate)
{
  const PetscReal L[3]{
    box.max[X] - box.min[X],
    box.max[Y] - box.min[Y],
    box.max[Z] - box.min[Z],
  };

  return Vector3R{
    velocity[X] * std::sin(
      2.0 * M_PI * wave_number[X] * coordinate[X] / L[X] + phase[X]),
    velocity[Y] * std::sin(
      2.0 * M_PI * wave_number[Y] * coordinate[Y] / L[Y] + phase[Y]),
    velocity[Z] * std::sin(
      2.0 * M_PI * wave_number[Z] * coordinate[Z] / L[Z] + phase[Z]),
  };
}
Vector3R MaxwellCosinePerturbation::operator()(const Vector3R& coordinate)
{
  static const PetscReal Lx = (box.max[X] - box.min[X]);
  static const PetscReal Ly = (box.max[Y] - box.min[Y]);
  static const PetscReal Lz = (box.max[Z] - box.min[Z]);

  Vector3R v_m{
    std::sin(2.0 * M_PI * random_01()) * temperature_momentum(params.Tx, params.m),
    std::sin(2.0 * M_PI * random_01()) * temperature_momentum(params.Ty, params.m),
    std::sin(2.0 * M_PI * random_01()) * temperature_momentum(params.Tz, params.m),
  };

  v_m /= std::sqrt(params.m * params.m + v_m.squared());

  Vector3R v_0{
    a[X] * sqrt(params.Tx / (params.m * mec2)),
    a[Y] * sqrt(params.Ty / (params.m * mec2)),
    a[Z] * sqrt(params.Tz / (params.m * mec2)),
  };

  v_m[X] += v_0[X] * std::cos(2.0 * M_PI * m[X] * coordinate[X] / Lx);
  v_m[Y] += v_0[Y] * std::cos(2.0 * M_PI * m[Y] * coordinate[Y] / Ly);
  v_m[Z] += v_0[Z] * std::cos(2.0 * M_PI * m[Z] * coordinate[Z] / Lz);

  return v_m;
}

Vector3R MaxwellShiftedSine::operator()(const Vector3R& coordinate)
{
  return sample_thermal_velocity(params)
    + sine_velocity_shift(box, velocity, wave_number, phase, coordinate);
}

KineticIonSoundMoments::KineticIonSoundMoments(
  const SortParameters& params, BoxGeometry box,
  PetscReal force_electric_amplitude, PetscReal omega_real, PetscReal gamma,
  const Vector3R& wave_number, const Vector3R& field_phase,
  const Vector3R& density_amplitude, const Vector3R& density_phase)
  : params(params), box(std::move(box)),
    force_electric_amplitude(force_electric_amplitude),
    omega_real(omega_real),
    gamma(gamma), wave_number(wave_number), field_phase(field_phase),
    density_amplitude(density_amplitude), density_phase(density_phase)
{
  if (!(params.n > 0.0) || !(params.m > 0.0) ||
      !std::isfinite(params.n) || !std::isfinite(params.q) ||
      !std::isfinite(params.m) ||
      !std::isfinite(force_electric_amplitude) ||
      !std::isfinite(omega_real) || !(gamma > 0.0) ||
      !std::isfinite(gamma))
    throw std::runtime_error(
      "KineticIonSoundMoments requires finite plasma parameters, n > 0, m > 0, and gamma > 0");

  const PetscReal temperatures[3] = {params.Tx, params.Ty, params.Tz};
  PetscInt active = 0;
  for (const Axis a : {X, Y, Z}) {
    if (temperatures[a] < 0.0 || !std::isfinite(temperatures[a]) ||
        !std::isfinite(wave_number[a]) ||
        !std::isfinite(field_phase[a]) ||
        !std::isfinite(density_amplitude[a]) ||
        !std::isfinite(density_phase[a]))
      throw std::runtime_error(
        "KineticIonSoundMoments requires finite mode parameters and non-negative temperatures");
    if (wave_number[a] != 0.0) {
      axis = a;
      ++active;
    }
    else if (density_amplitude[a] != 0.0) {
      throw std::runtime_error(
        "KineticIonSoundMoments density perturbation must use the active wave axis");
    }
  }
  if (active != 1)
    throw std::runtime_error(
      "KineticIonSoundMoments requires exactly one non-zero wave_number");

  const PetscReal length = this->box.max[axis] - this->box.min[axis];
  if (!(length > 0.0) || !std::isfinite(length))
    throw std::runtime_error(
      "KineticIonSoundMoments requires a finite non-empty box");
  if (!(std::abs(density_amplitude[axis]) < 1.0))
    throw std::runtime_error(
      "KineticIonSoundMoments requires |density_amplitude| < 1");

  equilibrium_variance =
    temperatures[axis] / (params.m * mec2);
  if (!(equilibrium_variance > 0.0) ||
      !std::isfinite(equilibrium_variance))
    throw std::runtime_error(
      "KineticIonSoundMoments requires positive parallel temperature");

  k_parallel = 2.0 * M_PI * wave_number[axis] / length;
  const std::complex<PetscReal> imaginary{0.0, 1.0};
  const std::complex<PetscReal> omega{omega_real, -gamma};
  const std::complex<PetscReal> density_phase_factor =
    std::exp(imaginary * density_phase[axis]);
  const std::complex<PetscReal> field_harmonic =
    force_electric_amplitude * std::exp(imaginary * field_phase[axis]);

  density_harmonic = -imaginary * params.n *
    density_amplitude[axis] * density_phase_factor;
  flux_harmonic = omega * density_harmonic / k_parallel;
  second_moment_harmonic =
    (omega * flux_harmonic -
      imaginary * (params.q / params.m) * params.n * field_harmonic) /
    k_parallel;

  const PetscReal density_min =
    params.n - std::abs(density_harmonic);
  const PetscReal second_moment_min =
    params.n * equilibrium_variance - std::abs(second_moment_harmonic);
  const PetscReal flux_max = std::abs(flux_harmonic);
  const PetscReal determinant_margin =
    density_min * second_moment_min - flux_max * flux_max;
  const PetscReal subluminal_margin = density_min - flux_max;
  if (!(density_min > 0.0) || !(second_moment_min > 0.0) ||
      !(determinant_margin > 0.0) || !(subluminal_margin > 0.0))
    throw std::runtime_error(std::format(
      "KineticIonSoundMoments positivity check failed for '{}': "
      "n_min={} (>0), P_min={} (>0), n_min*P_min-|M1|^2={} (>0), "
      "n_min-|M1|={} (>0), |M1|={}, |M2|={}, k={}. Regenerate the "
      "mode constants for the configured plasma, box, and grid with "
      "ion_sound.py --theory --grid-dz.",
      params.sort_name.empty() ? "<unnamed>" : params.sort_name,
      density_min, second_moment_min, determinant_margin,
      subluminal_margin, flux_max, std::abs(second_moment_harmonic),
      k_parallel));
}

void KineticIonSoundMoments::local_parallel_parameters(
  const Vector3R& coordinate, PetscReal& bulk_velocity,
  PetscReal& variance) const
{
  const PetscReal theta =
    k_parallel * (coordinate[axis] - box.min[axis]);
  const std::complex<PetscReal> phase_factor =
    std::polar<PetscReal>(1.0, theta);
  const PetscReal density =
    params.n + std::real(density_harmonic * phase_factor);
  const PetscReal flux = std::real(flux_harmonic * phase_factor);
  const PetscReal second_moment = params.n * equilibrium_variance +
    std::real(second_moment_harmonic * phase_factor);
  bulk_velocity = flux / density;
  variance = second_moment / density - bulk_velocity * bulk_velocity;

  if (!(density > 0.0) || !(variance > 0.0) ||
      !(std::abs(bulk_velocity) < 1.0) || !std::isfinite(variance))
    throw std::runtime_error(
      "KineticIonSoundMoments encountered invalid local moments");
}

Vector3R KineticIonSoundMoments::operator()(
  const Vector3R& coordinate)
{
  if (return_antithetic) {
    if ((coordinate - first_coordinate).abs_max() != 0.0)
      throw std::runtime_error(
        "KineticIonSoundMoments requires consecutive duplicate coordinates");
    return_antithetic = false;
    return second_velocity;
  }

  PetscReal bulk_velocity = 0.0;
  PetscReal variance = 0.0;
  local_parallel_parameters(coordinate, bulk_velocity, variance);
  const PetscReal parallel_scale =
    std::sqrt(variance / equilibrium_variance);

  constexpr std::size_t max_rejection_attempts = 1'000'000;
  for (std::size_t attempt = 0; attempt < max_rejection_attempts; ++attempt) {
    if (candidate_counter == std::numeric_limits<std::size_t>::max())
      throw std::runtime_error(
        "KineticIonSoundMoments exhausted its quiet sequence");

    Vector3R thermal_velocity = sample_maxwellian_velocity_quiet(
      params, ++candidate_counter);
    thermal_velocity[axis] *= parallel_scale;

    Vector3R first_velocity = thermal_velocity;
    first_velocity[axis] += bulk_velocity;
    second_velocity = -1.0 * thermal_velocity;
    second_velocity[axis] += bulk_velocity;
    if (std::isfinite(first_velocity.squared()) &&
        first_velocity.squared() < 1.0 &&
        std::isfinite(second_velocity.squared()) &&
        second_velocity.squared() < 1.0) {
      first_coordinate = coordinate;
      return_antithetic = true;
      return first_velocity;
    }
  }

  throw std::runtime_error(
    "KineticIonSoundMoments could not sample a subluminal pair");
}

Vector3R AngularMomentum::operator()(const Vector3R& coordinate)
{
  PetscReal x = coordinate.x() - center[X];
  PetscReal y = coordinate.y() - center[Y];
  PetscReal r = std::hypot(x, y);

  Vector3R temperature_moment{
    temperature_momentum(params.Tx, params.m),
    temperature_momentum(params.Ty, params.m),
    temperature_momentum(params.Tz, params.m),
  };

  if (std::isinf(1.0 / r))
    return Vector3R{0.0, 0.0, params.pz} + temperature_moment;

  return Vector3R{
    -params.px * (y / r) + temperature_moment[X],
    +params.py * (x / r) + temperature_moment[Y],
    +params.pz + temperature_moment[Z],
  };
}
