#include "particles_load.h"

#include "src/utils/random_generator.h"
#include "src/utils/utils.h"

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
}  // namespace

Vector3R CoordinateInBoxQuietSine::operator()()
{
  static const PetscReal L[3]{
    box.max[X] - box.min[X],
    box.max[Y] - box.min[Y],
    box.max[Z] - box.min[Z],
  };

  static constexpr std::size_t bases[3] = {2, 3, 5};

  // Halton index starts at 1 to keep u in (0, 1) on the very first call.
  const std::size_t i = counter + 1;
  std::size_t halton_slot = 0;

  Vector3R r;
  for (Axis a : {X, Y, Z}) {
    PetscReal u;
    if (amplitude[a] != 0.0 && wave_number[a] != 0.0)
      u = van_der_corput(i, bases[halton_slot++]);
    else
      u = random_01();
    r[a] = box.min[a] + u * L[a];
  }

  for (Axis a : {X, Y, Z}) {
    if (amplitude[a] == 0.0 || wave_number[a] == 0.0)
      continue;
    const PetscReal phase = 2.0 * M_PI * wave_number[a] * (r[a] - box.min[a]) / L[a];
    const PetscReal scale = amplitude[a] * L[a] / (2.0 * M_PI * wave_number[a]);
    r[a] += scale * (std::cos(phase) - 1.0);
  }

  ++counter;
  return r;
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
  static const PetscReal Lx = (box.max[X] - box.min[X]);
  static const PetscReal Ly = (box.max[Y] - box.min[Y]);
  static const PetscReal Lz = (box.max[Z] - box.min[Z]);

  Vector3R v_m{
    std::sin(2.0 * M_PI * random_01()) * temperature_momentum(params.Tx, params.m),
    std::sin(2.0 * M_PI * random_01()) * temperature_momentum(params.Ty, params.m),
    std::sin(2.0 * M_PI * random_01()) * temperature_momentum(params.Tz, params.m),
  };

  v_m /= std::sqrt(params.m * params.m + v_m.squared());

  v_m[X] += velocity[X] * std::sin(2.0 * M_PI * wave_number[X] * coordinate[X] / Lx);
  v_m[Y] += velocity[Y] * std::sin(2.0 * M_PI * wave_number[Y] * coordinate[Y] / Ly);
  v_m[Z] += velocity[Z] * std::sin(2.0 * M_PI * wave_number[Z] * coordinate[Z] / Lz);

  return v_m;
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
