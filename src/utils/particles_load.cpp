#include "particles_load.h"

#include "src/utils/random_generator.h"
#include "src/utils/utils.h"

#include <algorithm>
#include <array>
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

Vector3R CoordinateInBoxSineDensity::operator()()
{
  static const PetscReal Lx = (box.max[X] - box.min[X]);
  static const PetscReal Ly = (box.max[Y] - box.min[Y]);
  static const PetscReal Lz = (box.max[Z] - box.min[Z]);

  const PetscReal max_density = 1.0 +
    std::abs(amplitude[X]) + std::abs(amplitude[Y]) + std::abs(amplitude[Z]);

  while (true) {
    Vector3R r{
      box.min[X] + random_01() * Lx,
      box.min[Y] + random_01() * Ly,
      box.min[Z] + random_01() * Lz,
    };

    PetscReal density = 1.0
      + amplitude[X] * std::sin(2.0 * M_PI * wave_number[X] * (r[X] - box.min[X]) / Lx)
      + amplitude[Y] * std::sin(2.0 * M_PI * wave_number[Y] * (r[Y] - box.min[Y]) / Ly)
      + amplitude[Z] * std::sin(2.0 * M_PI * wave_number[Z] * (r[Z] - box.min[Z]) / Lz);

    if (random_01() * max_density < density)
      return r;
  }
}

namespace {
// A Halton phase-space load needs a different prime base for every dimension.
// Reusing a coordinate base for a velocity dimension creates a deterministic
// position-temperature correlation which antithetic velocity pairs do not
// cancel. Keep the coordinate and Maxwellian velocity allocations together so
// that future changes cannot silently reintroduce an overlap.
constexpr std::array<std::size_t, 3> quiet_coordinate_bases{2, 3, 5};
constexpr std::array<std::size_t, 3> quiet_velocity_radius_bases{7, 11, 13};
constexpr std::array<std::size_t, 3> quiet_velocity_phase_bases{17, 19, 23};

consteval bool bases_are_disjoint(const auto& lhs, const auto& rhs)
{
  for (const auto a : lhs)
    for (const auto b : rhs)
      if (a == b)
        return false;
  return true;
}

static_assert(
  bases_are_disjoint(quiet_coordinate_bases, quiet_velocity_radius_bases) &&
  bases_are_disjoint(quiet_coordinate_bases, quiet_velocity_phase_bases) &&
  bases_are_disjoint(quiet_velocity_radius_bases, quiet_velocity_phase_bases));

// Van der Corput sequence in given `base`: reverse the digits of `i` in
// base `b` and read the result as a fraction in [0, 1). For composite
// generators (Halton) different prime bases are used per axis so the
// projections stay decorrelated.
PetscReal van_der_corput(std::size_t i, std::size_t base)
{
  PetscReal q = 0.0;
  PetscReal bk = 1.0 / static_cast<PetscReal>(base);
  while (i > 0) {
    q += static_cast<PetscReal>(i % base) * bk;
    i /= base;
    bk /= static_cast<PetscReal>(base);
  }
  return q;
}

// Inverse CDF of the cosine-hump density p(s) = 0.5 (1 + cos(pi s)) on
// s in [-1, 1]; CDF F(s) = 0.5 (s + 1) + sin(pi s) / (2 pi). Safeguarded
// Newton: F' = p vanishes at s = +-1, so a plain Newton step overshoots
// for u near 0 or 1 (delta blows up while F - u is tiny). We keep a
// bracket [s_lo, s_hi] where F(s_lo) <= u <= F(s_hi), accept the Newton
// step only if it stays strictly inside the bracket, and fall back to
// bisection otherwise. Converges for every u in [0, 1].
PetscReal cosine_hump_inv_cdf(PetscReal u)
{
  PetscReal s_lo = -1.0;
  PetscReal s_hi = +1.0;
  PetscReal s = 2.0 * u - 1.0;

  for (int it = 0; it < 64; ++it) {
    const PetscReal F  = 0.5 * (s + 1.0) + std::sin(M_PI * s) / (2.0 * M_PI);
    const PetscReal Fp = 0.5 * (1.0 + std::cos(M_PI * s));

    if (F < u) s_lo = s;
    else       s_hi = s;

    PetscReal s_new;
    if (Fp > 1.0e-12) {
      s_new = s - (F - u) / Fp;
      if (s_new <= s_lo || s_new >= s_hi)
        s_new = 0.5 * (s_lo + s_hi);
    }
    else {
      s_new = 0.5 * (s_lo + s_hi);
    }

    if (std::abs(s_new - s) < 1.0e-14)
      return s_new;
    s = s_new;
  }
  return s;
}

PetscReal sine_density_inv_cdf(
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

// Factor `n` into three positive integers out[0] <= out[1] <= out[2] with
// out[0]*out[1]*out[2] == n, choosing the triple closest to a cube (minimal
// out[2] - out[0]). Falls back to (1, 1, n) for primes. Used to spread the
// `Np` particles of a cell over a balanced sub-lattice.
void factor_three(PetscInt n, PetscInt out[3])
{
  out[0] = 1;
  out[1] = 1;
  out[2] = n;
  PetscInt best_spread = n - 1;

  const auto a_max = static_cast<PetscInt>(std::cbrt(static_cast<double>(n)));
  for (PetscInt a = 1; a <= a_max; ++a) {
    if (n % a != 0)
      continue;
    const PetscInt m = n / a;
    const auto b_max = static_cast<PetscInt>(std::sqrt(static_cast<double>(m)));
    for (PetscInt b = a; b <= b_max; ++b) {
      if (m % b != 0)
        continue;
      const PetscInt c = m / b;  // a <= b <= c
      if (c - a < best_spread) {
        best_spread = c - a;
        out[0] = a;
        out[1] = b;
        out[2] = c;
      }
    }
  }
}

Vector3R quiet_sine_coordinate(const BoxGeometry& box,
  const Vector3R& amplitude, const Vector3R& wave_number,
  const Vector3R& phase, std::size_t index, bool quiet_all_axes = false)
{
  const PetscReal L[3]{
    box.max[X] - box.min[X],
    box.max[Y] - box.min[Y],
    box.max[Z] - box.min[Z],
  };

  std::size_t halton_slot = 0;

  Vector3R r;
  for (Axis a : {X, Y, Z}) {
    PetscReal u;
    if (quiet_all_axes)
      u = van_der_corput(
        index, quiet_coordinate_bases[static_cast<std::size_t>(a)]);
    else if (amplitude[a] != 0.0 && wave_number[a] != 0.0)
      u = van_der_corput(index, quiet_coordinate_bases[halton_slot++]);
    else
      u = random_01();
    r[a] = box.min[a] + u * L[a];
  }

  for (Axis a : {X, Y, Z}) {
    if (amplitude[a] == 0.0 || wave_number[a] == 0.0)
      continue;
    const PetscReal arg =
      2.0 * M_PI * wave_number[a] * (r[a] - box.min[a]) / L[a] + phase[a];
    const PetscReal scale = amplitude[a] * L[a] / (2.0 * M_PI * wave_number[a]);
    r[a] += scale * (std::cos(arg) - std::cos(phase[a]));
  }

  return r;
}
}  // namespace

Vector3R CoordinateInBoxQuietSine::operator()()
{
  // Halton index starts at 1 to keep u in (0, 1) on the very first call.
  Vector3R r = quiet_sine_coordinate(
    box, amplitude, wave_number, phase, counter + 1);
  ++counter;
  return r;
}

Vector3R CoordinateInBoxQuietSinePaired::operator()()
{
  if (return_second) {
    return_second = false;
    return paired_coordinate;
  }

  // Halton index starts at 1 to keep u in (0, 1) on the first pair.
  paired_coordinate = quiet_sine_coordinate(
    box, amplitude, wave_number, phase, pair_counter + 1, true);
  ++pair_counter;
  return_second = true;
  return paired_coordinate;
}

CoordinateInBoxQuietSineExactPaired::CoordinateInBoxQuietSineExactPaired(
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
        "CoordinateInBoxQuietSineExactPaired requires amplitude and wave_number together");
    if (std::abs(amplitude[a]) >= 1.0)
      throw std::runtime_error(
        "CoordinateInBoxQuietSineExactPaired requires |amplitude| < 1");
    const PetscReal rounded_mode = std::round(wave_number[a]);
    if (rounded_mode <= 0.0 ||
        std::abs(wave_number[a] - rounded_mode) > 1.0e-12)
      throw std::runtime_error(
        "CoordinateInBoxQuietSineExactPaired requires a positive integer wave_number");
    axis = a;
    ++active;
  }
  if (active != 1)
    throw std::runtime_error(
      "CoordinateInBoxQuietSineExactPaired requires exactly one active axis");
}

Vector3R CoordinateInBoxQuietSineExactPaired::operator()()
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
          "CoordinateInBoxQuietSineExactPaired exhausted its active-axis lattice");
      u = (static_cast<PetscReal>(pair_counter) + 0.5) /
        static_cast<PetscReal>(active_axis_lattice_pairs);
    }
    else {
      u = van_der_corput(
        index, quiet_coordinate_bases[static_cast<std::size_t>(a)]);
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

Vector3R CoordinateInBoxDisplacedSine::operator()()
{
  static const PetscReal L[3]{
    box.max[X] - box.min[X],
    box.max[Y] - box.min[Y],
    box.max[Z] - box.min[Z],
  };

  Vector3R r{
    box.min[X] + random_01() * L[X],
    box.min[Y] + random_01() * L[Y],
    box.min[Z] + random_01() * L[Z],
  };

  // Per-axis displacement T(r) - r = (A L / 2 pi k) (cos(2 pi k r / L) - 1),
  // chosen so that uniform `r` mapped by T has density 1 + A sin(2 pi k r / L)
  // to O(A^2). Axes with amplitude=0 or wave_number=0 are left untouched.
  for (Axis a : {X, Y, Z}) {
    if (amplitude[a] == 0.0 || wave_number[a] == 0.0)
      continue;
    const PetscReal phase = 2.0 * M_PI * wave_number[a] * (r[a] - box.min[a]) / L[a];
    const PetscReal scale = amplitude[a] * L[a] / (2.0 * M_PI * wave_number[a]);
    r[a] += scale * (std::cos(phase) - 1.0);
  }

  return r;
}

Vector3R CoordinateInBoxQuiet::operator()()
{
  const PetscReal L[3]{
    box.max[X] - box.min[X],
    box.max[Y] - box.min[Y],
    box.max[Z] - box.min[Z],
  };

  // Number of grid cells covered by the box along each axis.
  PetscInt Nc[3];
  for (Axis a : {X, Y, Z})
    Nc[a] = std::max<PetscInt>(1,
      static_cast<PetscInt>(std::llround(L[a] / Dx[a])));

  // Balanced per-cell sub-lattice np[X] * np[Y] * np[Z] == Np, so the global
  // lattice has G[a] = Nc[a] * np[a] nodes and each cell owns exactly np[a]
  // of them per axis.
  PetscInt np[3];
  factor_three(Np, np);

  const PetscInt G[3]{Nc[X] * np[X], Nc[Y] * np[Y], Nc[Z] * np[Z]};

  // Mixed-radix decomposition of the global particle index into node indices.
  std::size_t idx = counter;
  PetscInt g[3];
  g[X] = static_cast<PetscInt>(idx % G[X]); idx /= G[X];
  g[Y] = static_cast<PetscInt>(idx % G[Y]); idx /= G[Y];
  g[Z] = static_cast<PetscInt>(idx % G[Z]);

  Vector3R r;
  for (Axis a : {X, Y, Z})
    r[a] = box.min[a] + (g[a] + 0.5) / static_cast<PetscReal>(G[a]) * L[a];

  ++counter;
  return r;
}

Vector3R CoordinateInCylinderCosineHump::operator()()
{
  while (true) {
    PetscReal r = cyl.radius * std::sqrt(random_01());
    PetscReal phi = 2.0 * M_PI * random_01();
    PetscReal z = cyl.height * (random_01() - 0.5);

    PetscReal w = 0.5 * (1.0 + std::cos(M_PI * r / cyl.radius));

    if (random_01() < w)
      return Vector3R{
        cyl.center[X] + r * std::cos(phi),
        cyl.center[Y] + r * std::sin(phi),
        cyl.center[Z] + z,
      };
  }
}

Vector3R CoordinateInBoxCosineHump::operator()()
{
  const std::size_t i = counter + 1;
  ++counter;

  // Base 2 drives the hump axis (low-discrepancy on the non-trivial CDF),
  // bases 3 and 5 drive the two uniform axes.
  const PetscReal u_hump = van_der_corput(i, 2);
  std::size_t uniform_base_slot = 0;
  static constexpr std::size_t uniform_bases[2] = {3, 5};

  Vector3R r;
  for (Axis a : {X, Y, Z}) {
    if (a == axis) {
      const PetscReal s = cosine_hump_inv_cdf(u_hump);
      const PetscReal half_L = 0.5 * (box.max[a] - box.min[a]);
      const PetscReal center_a = 0.5 * (box.max[a] + box.min[a]);
      r[a] = center_a + s * half_L;
    }
    else {
      const PetscReal u = van_der_corput(i, uniform_bases[uniform_base_slot++]);
      r[a] = box.min[a] + u * (box.max[a] - box.min[a]);
    }
  }

  return r;
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

namespace {
Vector3R sample_thermal_velocity(const SortParameters& params)
{
  Vector3R result{
    std::sin(2.0 * M_PI * random_01()) * temperature_momentum(params.Tx, params.m),
    std::sin(2.0 * M_PI * random_01()) * temperature_momentum(params.Ty, params.m),
    std::sin(2.0 * M_PI * random_01()) * temperature_momentum(params.Tz, params.m),
  };
  result /= std::sqrt(params.m * params.m + result.squared());
  return result;
}

Vector3R sample_thermal_velocity_quiet(
  const SortParameters& params, std::size_t index)
{
  // Bases are also disjoint from the coordinate bases used by the paired
  // quiet loader. Starting at index 1 keeps every radial sample strictly
  // inside (0, 1), so log(u) is finite.
  const PetscReal temperatures[3] = {params.Tx, params.Ty, params.Tz};

  Vector3R result;
  for (Axis a : {X, Y, Z}) {
    const auto slot = static_cast<std::size_t>(a);
    const PetscReal radius =
      std::sqrt(-2.0 * (temperatures[slot] * params.m / mec2)
        * std::log(van_der_corput(
          index, quiet_velocity_radius_bases[slot])));
    const PetscReal phase =
      2.0 * M_PI * van_der_corput(index, quiet_velocity_phase_bases[slot]);
    result[a] = radius * std::sin(phase);
  }

  result /= std::sqrt(params.m * params.m + result.squared());
  return result;
}

Vector3R sample_maxwellian_velocity_quiet(
  const SortParameters& params, std::size_t index)
{
  const PetscReal temperatures[3] = {params.Tx, params.Ty, params.Tz};

  Vector3R result;
  for (Axis a : {X, Y, Z}) {
    const auto slot = static_cast<std::size_t>(a);
    const PetscReal sigma =
      std::sqrt(temperatures[slot] / (params.m * mec2));
    const PetscReal radius = std::sqrt(-2.0 * std::log(van_der_corput(
      index, quiet_velocity_radius_bases[slot])));
    const PetscReal phase =
      2.0 * M_PI * van_der_corput(index, quiet_velocity_phase_bases[slot]);
    result[a] = sigma * radius * std::sin(phase);
  }
  return result;
}

Vector3R sine_velocity_shift(const BoxGeometry& box, const Vector3R& velocity,
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
}  // namespace

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

Vector3R MaxwellShiftedSineQuiet::operator()(const Vector3R& coordinate)
{
  Vector3R result;
  if (!return_antithetic) {
    thermal_velocity =
      sample_thermal_velocity_quiet(params, pair_counter + 1);
    result = thermal_velocity;
    return_antithetic = true;
  }
  else {
    result = -1.0 * thermal_velocity;
    return_antithetic = false;
    ++pair_counter;
  }

  return result
    + sine_velocity_shift(box, velocity, wave_number, phase, coordinate);
}

MaxwellianVelocityQuiet::MaxwellianVelocityQuiet(
  const SortParameters& params)
  : params(params)
{
  if (!(params.m > 0.0) || !std::isfinite(params.m))
    throw std::runtime_error(
      "MaxwellianVelocityQuiet requires a finite positive mass");

  for (const PetscReal temperature : {params.Tx, params.Ty, params.Tz}) {
    if (temperature < 0.0 || !std::isfinite(temperature))
      throw std::runtime_error(
        "MaxwellianVelocityQuiet requires finite non-negative temperatures");
  }
}

Vector3R MaxwellianVelocityQuiet::operator()(
  const Vector3R& /* coordinate */)
{
  if (return_antithetic) {
    return_antithetic = false;
    return -1.0 * thermal_velocity;
  }

  constexpr std::size_t max_rejection_attempts = 1'000'000;
  for (std::size_t attempt = 0; attempt < max_rejection_attempts; ++attempt) {
    if (candidate_counter == std::numeric_limits<std::size_t>::max())
      throw std::runtime_error(
        "MaxwellianVelocityQuiet exhausted its quiet sequence");

    thermal_velocity = sample_maxwellian_velocity_quiet(
      params, ++candidate_counter);
    if (std::isfinite(thermal_velocity.squared()) &&
        thermal_velocity.squared() < 1.0) {
      return_antithetic = true;
      return thermal_velocity;
    }
  }

  throw std::runtime_error(
    "MaxwellianVelocityQuiet could not sample a subluminal velocity");
}

KineticIonSoundMomentsQuiet::KineticIonSoundMomentsQuiet(
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
      "KineticIonSoundMomentsQuiet requires finite plasma parameters, n > 0, m > 0, and gamma > 0");

  const PetscReal temperatures[3] = {params.Tx, params.Ty, params.Tz};
  PetscInt active = 0;
  for (const Axis a : {X, Y, Z}) {
    if (temperatures[a] < 0.0 || !std::isfinite(temperatures[a]) ||
        !std::isfinite(wave_number[a]) ||
        !std::isfinite(field_phase[a]) ||
        !std::isfinite(density_amplitude[a]) ||
        !std::isfinite(density_phase[a]))
      throw std::runtime_error(
        "KineticIonSoundMomentsQuiet requires finite mode parameters and non-negative temperatures");
    if (wave_number[a] != 0.0) {
      axis = a;
      ++active;
    }
    else if (density_amplitude[a] != 0.0) {
      throw std::runtime_error(
        "KineticIonSoundMomentsQuiet density perturbation must use the active wave axis");
    }
  }
  if (active != 1)
    throw std::runtime_error(
      "KineticIonSoundMomentsQuiet requires exactly one non-zero wave_number");

  const PetscReal length = this->box.max[axis] - this->box.min[axis];
  if (!(length > 0.0) || !std::isfinite(length))
    throw std::runtime_error(
      "KineticIonSoundMomentsQuiet requires a finite non-empty box");
  if (!(std::abs(density_amplitude[axis]) < 1.0))
    throw std::runtime_error(
      "KineticIonSoundMomentsQuiet requires |density_amplitude| < 1");

  equilibrium_variance =
    temperatures[axis] / (params.m * mec2);
  if (!(equilibrium_variance > 0.0) ||
      !std::isfinite(equilibrium_variance))
    throw std::runtime_error(
      "KineticIonSoundMomentsQuiet requires positive parallel temperature");

  k_parallel = 2.0 * M_PI * wave_number[axis] / length;
  const std::complex<PetscReal> imaginary{0.0, 1.0};
  const std::complex<PetscReal> omega{omega_real, -gamma};
  const std::complex<PetscReal> density_phase_factor =
    std::exp(imaginary * density_phase[axis]);
  const std::complex<PetscReal> field_harmonic =
    force_electric_amplitude * std::exp(imaginary * field_phase[axis]);

  // A configured sine a*sin(kz+phi) has complex harmonic
  // -i*a*exp(i*phi) under the Re[hat{a} exp(ikz)] convention.
  density_harmonic = -imaginary * params.n *
    density_amplitude[axis] * density_phase_factor;
  flux_harmonic = omega * density_harmonic / k_parallel;
  second_moment_harmonic =
    (omega * flux_harmonic -
      imaginary * (params.q / params.m) * params.n * field_harmonic) /
    k_parallel;

  // Strict all-phase sufficient bounds.  They are deliberately conservative:
  // n >= n_min, P >= P_min, |Gamma| <= flux_max imply
  // n*P-Gamma^2 > 0 and |Gamma/n| < 1 at every coordinate.
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
      "KineticIonSoundMomentsQuiet positivity check failed for '{}': "
      "n_min={} (>0), P_min={} (>0), n_min*P_min-|M1|^2={} (>0), "
      "n_min-|M1|={} (>0), |M1|={}, |M2|={}, k={}. Regenerate the "
      "mode constants for the configured plasma, box, and grid with "
      "ion_sound.py --theory --grid-dz.",
      params.sort_name.empty() ? "<unnamed>" : params.sort_name,
      density_min, second_moment_min, determinant_margin,
      subluminal_margin, flux_max, std::abs(second_moment_harmonic),
      k_parallel));
}

void KineticIonSoundMomentsQuiet::local_parallel_parameters(
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
      "KineticIonSoundMomentsQuiet encountered invalid local moments");
}

Vector3R KineticIonSoundMomentsQuiet::operator()(
  const Vector3R& coordinate)
{
  if (return_antithetic) {
    if ((coordinate - first_coordinate).abs_max() != 0.0)
      throw std::runtime_error(
        "KineticIonSoundMomentsQuiet requires consecutive duplicate coordinates");
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
        "KineticIonSoundMomentsQuiet exhausted its quiet sequence");

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
    "KineticIonSoundMomentsQuiet could not sample a subluminal pair");
}

KineticIonSoundQuiet::KineticIonSoundQuiet(const SortParameters& params,
  BoxGeometry box, PetscReal electric_amplitude, PetscReal omega_real,
  PetscReal gamma, const Vector3R& wave_number,
  const Vector3R& field_phase, PetscReal velocity_cutoff_vT,
  PetscReal velocity_abs_max)
  : params(params), box(std::move(box)),
    electric_amplitude(electric_amplitude), omega_real(omega_real),
    gamma(gamma), wave_number(wave_number), field_phase(field_phase),
    velocity_cutoff_vT(velocity_cutoff_vT),
    velocity_abs_max(velocity_abs_max)
{
  PetscInt active = 0;
  for (Axis a : {X, Y, Z}) {
    if (wave_number[a] == 0.0)
      continue;
    axis = a;
    ++active;
  }
  if (active != 1)
    throw std::runtime_error(
      "KineticIonSoundQuiet requires exactly one non-zero wave_number");
  if (!(gamma > 0.0) || !(velocity_cutoff_vT > 0.0) ||
      !(velocity_abs_max > 0.0 && velocity_abs_max < 1.0))
    throw std::runtime_error(
      "KineticIonSoundQuiet requires gamma > 0, velocity_cutoff_vT > 0, and 0 < velocity_abs_max < 1");

  const PetscReal temperatures[3] = {params.Tx, params.Ty, params.Tz};
  vT_parallel =
    std::sqrt(temperatures[axis] / (params.m * mec2));
  if (!(vT_parallel > 0.0))
    throw std::runtime_error(
      "KineticIonSoundQuiet requires positive parallel temperature");

  const PetscReal length = box.max[axis] - box.min[axis];
  if (!(length > 0.0))
    throw std::runtime_error("KineticIonSoundQuiet requires a non-empty box");
  k_parallel = 2.0 * M_PI * wave_number[axis] / length;
  v_parallel_max =
    std::min(velocity_cutoff_vT * vT_parallel, velocity_abs_max);

  build_inverse_cdf();
}

void KineticIonSoundQuiet::build_inverse_cdf()
{
  constexpr PetscInt phase_bins = 512;
  constexpr PetscInt quantile_bins = 4096;
  constexpr PetscInt velocity_bins = 8192;
  inverse_cdf = std::make_shared<std::vector<PetscReal>>(
    static_cast<std::size_t>(phase_bins) * (quantile_bins + 1));

  const PetscReal coefficient =
    params.q * electric_amplitude /
    (params.m * vT_parallel * vT_parallel);
  const PetscReal dv = 2.0 * v_parallel_max / velocity_bins;
  std::vector<PetscReal> velocity(velocity_bins + 1);
  std::vector<PetscReal> pdf(velocity_bins + 1);
  std::vector<PetscReal> cdf(velocity_bins + 1);
  for (PetscInt iv = 0; iv <= velocity_bins; ++iv)
    velocity[iv] = -v_parallel_max + iv * dv;

  for (PetscInt ip = 0; ip < phase_bins; ++ip) {
    const PetscReal theta =
      2.0 * M_PI * static_cast<PetscReal>(ip) / phase_bins;
    for (PetscInt iv = 0; iv <= velocity_bins; ++iv) {
      const PetscReal v = velocity[iv];
      const PetscReal detuning = omega_real - k_parallel * v;
      const PetscReal denominator = detuning * detuning + gamma * gamma;
      const PetscReal h = -coefficient * v *
        (gamma * std::cos(theta) + detuning * std::sin(theta)) /
        denominator;
      const PetscReal maxwell =
        std::exp(-0.5 * POW2(v / vT_parallel));
      pdf[iv] = maxwell * std::max<PetscReal>(0.0, 1.0 + h);
    }

    cdf[0] = 0.0;
    for (PetscInt iv = 1; iv <= velocity_bins; ++iv)
      cdf[iv] = cdf[iv - 1] + 0.5 * dv * (pdf[iv - 1] + pdf[iv]);
    const PetscReal normalization = cdf.back();
    if (!(normalization > 0.0))
      throw std::runtime_error(
        "KineticIonSoundQuiet has zero conditional-PDF normalization");
    for (auto& value : cdf)
      value /= normalization;

    const std::size_t offset =
      static_cast<std::size_t>(ip) * (quantile_bins + 1);
    (*inverse_cdf)[offset] = -v_parallel_max;
    (*inverse_cdf)[offset + quantile_bins] = v_parallel_max;
    PetscInt iv = 1;
    for (PetscInt iq = 1; iq < quantile_bins; ++iq) {
      // Chebyshev--Lobatto quantiles cluster table entries at q=0 and q=1.
      // A uniform q grid makes the first segment connect the hard velocity
      // cutoff directly to Q(1/Nq); linear interpolation then turns the tiny
      // Maxwellian tail into an artificial flat population.  Here
      // q_1 ~= pi^2/(4 Nq^2), so fewer than one marker of the ex10 load lies
      // in either unresolved endpoint segment.
      const PetscReal quantile = 0.5 * (1.0 - std::cos(
        M_PI * static_cast<PetscReal>(iq) / quantile_bins));
      while (iv < velocity_bins && cdf[iv] < quantile)
        ++iv;
      const PetscReal dcdf = cdf[iv] - cdf[iv - 1];
      const PetscReal fraction = dcdf > 0.0
        ? (quantile - cdf[iv - 1]) / dcdf : 0.0;
      (*inverse_cdf)[offset + iq] =
        velocity[iv - 1] + fraction * (velocity[iv] - velocity[iv - 1]);
    }
  }
}

PetscReal KineticIonSoundQuiet::sample_parallel(
  const Vector3R& coordinate, PetscReal quantile) const
{
  constexpr PetscInt phase_bins = 512;
  constexpr PetscInt quantile_bins = 4096;
  PetscReal theta =
    k_parallel * (coordinate[axis] - box.min[axis]) + field_phase[axis];
  theta = std::fmod(theta, 2.0 * M_PI);
  if (theta < 0.0)
    theta += 2.0 * M_PI;

  const PetscReal phase_position = theta * phase_bins / (2.0 * M_PI);
  const PetscInt phase0 =
    static_cast<PetscInt>(std::floor(phase_position)) % phase_bins;
  const PetscInt phase1 = (phase0 + 1) % phase_bins;
  const PetscReal phase_fraction = phase_position - std::floor(phase_position);

  // Invert q(j) = [1-cos(pi*j/Nq)]/2 used by build_inverse_cdf().
  const PetscReal q = std::clamp(quantile, 0.0, 1.0);
  const PetscReal q_position = quantile_bins / M_PI *
    std::acos(std::clamp(1.0 - 2.0 * q, -1.0, 1.0));
  const PetscInt q0 = std::min<PetscInt>(
    static_cast<PetscInt>(std::floor(q_position)), quantile_bins - 1);
  const PetscInt q1 = q0 + 1;
  const PetscReal q_fraction = q_position - q0;
  const auto at = [&](PetscInt ip, PetscInt iq) {
    return (*inverse_cdf)[
      static_cast<std::size_t>(ip) * (quantile_bins + 1) + iq];
  };
  const PetscReal v0 = at(phase0, q0) +
    q_fraction * (at(phase0, q1) - at(phase0, q0));
  const PetscReal v1 = at(phase1, q0) +
    q_fraction * (at(phase1, q1) - at(phase1, q0));
  return v0 + phase_fraction * (v1 - v0);
}

Vector3R KineticIonSoundQuiet::sample_perpendicular_pair(
  PetscReal v_parallel_first, PetscReal v_parallel_second)
{
  Axis transverse[2];
  PetscInt slot = 0;
  for (Axis a : {X, Y, Z})
    if (a != axis)
      transverse[slot++] = a;

  const PetscReal temperatures[3] = {params.Tx, params.Ty, params.Tz};
  while (true) {
    const std::size_t index = ++perpendicular_counter;
    Vector3R result{};
    for (PetscInt j = 0; j < 2; ++j) {
      const Axis a = transverse[j];
      const std::size_t radius_base = j == 0 ? 29 : 37;
      const std::size_t phase_base = j == 0 ? 31 : 41;
      const PetscReal sigma =
        std::sqrt(temperatures[a] / (params.m * mec2));
      result[a] = sigma *
        std::sqrt(-2.0 * std::log(van_der_corput(index, radius_base))) *
        std::cos(2.0 * M_PI * van_der_corput(index, phase_base));
    }
    const PetscReal v_perp2 = result.squared();
    if (v_perp2 + v_parallel_first * v_parallel_first < 1.0 &&
        v_perp2 + v_parallel_second * v_parallel_second < 1.0)
      return result;
  }
}

Vector3R KineticIonSoundQuiet::operator()(const Vector3R& coordinate)
{
  if (return_second) {
    return_second = false;
    return second_velocity;
  }

  const PetscReal quantile = van_der_corput(++pair_counter, 17);
  const PetscReal v_parallel_first =
    sample_parallel(coordinate, quantile);
  const PetscReal v_parallel_second =
    sample_parallel(coordinate, 1.0 - quantile);
  const Vector3R v_perp =
    sample_perpendicular_pair(v_parallel_first, v_parallel_second);

  Vector3R first_velocity = v_perp;
  first_velocity[axis] = v_parallel_first;
  second_velocity = -1.0 * v_perp;
  second_velocity[axis] = v_parallel_second;
  return_second = true;
  return first_velocity;
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
