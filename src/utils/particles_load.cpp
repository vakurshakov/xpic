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
  PetscReal r = cyl.radius * sqrt(random_01());
  PetscReal phi = 2.0 * M_PI * random_01();

  return Vector3R{
    cyl.center[X] + r * cos(phi),
    cyl.center[Y] + r * sin(phi),
    cyl.center[Z] + cyl.height * (random_01() - 0.5),
  };
}

KineticIonSoundCoordinates::KineticIonSoundCoordinates(BoxGeometry box,
  const Vector3R& amplitude, const Vector3R& wave_number, const Vector3R& phase,
  std::size_t active_axis_lattice_pairs)
  : box(box),
    n_pa(amplitude),
    n_pm(wave_number),
    n_pp(phase),
    pairs_total_number(active_axis_lattice_pairs)
{
  PetscInt active = 0;

  for (Axis a : {X, Y, Z}) {
    if (n_pa[a] == 0 && n_pm[a] == 0)
      continue;

    PetscCheckAbort(n_pa[a] != 0 && n_pm[a] != 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "KineticIonSoundCoordinates requires amplitude and wave_number together");

    PetscCheckAbort(abs(n_pa[a]) < 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "KineticIonSoundCoordinates requires |amplitude| < 1");

    PetscReal round_pm = round(n_pm[a]);
    PetscCheckAbort(round_pm > 0 && abs(n_pm[a] - round_pm) <= 1e-12, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "KineticIonSoundCoordinates requires a positive integer wave_number");

    axis = a;
    active++;
  }

  PetscCheckAbort(active == 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
    "KineticIonSoundCoordinates requires exactly one active axis");
}

/**
 * @details
 * Halton quasi-Monte Carlo generator for low-discrepancy phase-space sampling.
 * Uses Van der Corput sequences with different prime bases per dimension to
 * produce decorrelated projections. Coordinate and velocity bases must be
 * disjoint to avoid position-velocity correlations that antithetic pairing
 * cannot cancel.
 */
namespace {

constexpr std::array<std::size_t, 3> coordinate_bases{2, 3, 5};
constexpr std::array<std::size_t, 3> velocity_radius_bases{7, 11, 13};
constexpr std::array<std::size_t, 3> velocity_phase_bases{17, 19, 23};

consteval bool bases_are_disjoint(const auto& lhs, const auto& rhs)
{
  for (const auto a : lhs)
    for (const auto b : rhs)
      if (a == b)
        return false;
  return true;
}

static_assert(  //
  bases_are_disjoint(coordinate_bases, velocity_radius_bases) &&
  bases_are_disjoint(coordinate_bases, velocity_phase_bases) &&
  bases_are_disjoint(velocity_radius_bases, velocity_phase_bases));

/// Reverts digits of `i` in `base`, @returns the result as fraction in [0,1).
PetscReal van_der_corput(std::size_t i, std::size_t base)
{
  PetscReal q = 0;
  PetscReal bk = 1.0 / static_cast<PetscReal>(base);
  while (i > 0) {
    q += static_cast<PetscReal>(i % base) * bk;
    i /= base;
    bk /= static_cast<PetscReal>(base);
  }
  return q;
}

}  // namespace

PetscReal sine_density_inv_cdf(
  PetscReal u, PetscReal amplitude, PetscReal mode, PetscReal phase)
{
  PetscReal x = u;
  PetscReal x_lo = 0.0;
  PetscReal x_hi = 1.0;

  PetscReal k = (2.0 * M_PI * mode);
  PetscReal k_inv = 1.0 / k;

  for (int it = 0; it < 64; it++) {
    PetscReal arg = k * x + phase;
    PetscReal cdf = x + amplitude * k_inv * (cos(phase) - cos(arg));
    PetscReal pdf = 1.0 + amplitude * sin(arg);

    ((cdf < u) ? x_lo : x_hi) = x;

    PetscReal x_new = x - (cdf - u) / pdf;

    if (x_new <= x_lo || x_new >= x_hi)
      x_new = 0.5 * (x_lo + x_hi);

    if (abs(x_new - x) < 16.0 * std::numeric_limits<PetscReal>::epsilon())
      return x_new;

    x = x_new;
  }
  return x;
}


Vector3R KineticIonSoundCoordinates::operator()()
{
  if (return_second_coordinate) {
    return_second_coordinate = false;
    return paired_coordinate;
  }

  for (Axis a : {X, Y, Z}) {
    PetscReal rand_x;

    if (a == axis) {
      /// @note pairs_total_number (active_axis_lattice_pairs) is always positive (?)
      PetscCheckAbort(pair_number < pairs_total_number, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
        "KineticIonSoundCoordinates exhausted its active-axis lattice");

      rand_x = ((PetscReal)pair_number + 0.5) / pairs_total_number;
      rand_x = sine_density_inv_cdf(rand_x, n_pa[a], n_pm[a], n_pp[a]);
    }
    else {
      rand_x = van_der_corput(pair_number + 1, coordinate_bases[a]);
    }

    paired_coordinate[a] = box.min[a] + rand_x * (box.max[a] - box.min[a]);
  }

  pair_number++;
  return_second_coordinate = true;
  return paired_coordinate;
}

Vector3R CoordinateInBoxCosineHump::operator()()
{
  PetscReal center = 0.5 * (box.min[axis] + box.max[axis]);
  PetscReal half_length = 0.5 * (box.max[axis] - box.min[axis]);

  while (true) {
    Vector3R coordinate{
      box.min[X] + random_01() * (box.max[X] - box.min[X]),
      box.min[Y] + random_01() * (box.max[Y] - box.min[Y]),
      box.min[Z] + random_01() * (box.max[Z] - box.min[Z]),
    };

    PetscReal normalized = (coordinate[axis] - center) / half_length;
    PetscReal weight = 0.5 * (1.0 + cos(M_PI * normalized));

    if (random_01() < weight)
      return coordinate;
  }
}

Vector3R CoordinateOnAnnulus::operator()()
{
  PetscReal r = sqrt(POW2(ann.inner_r) +  //
    (POW2(ann.outer_r) - POW2(ann.inner_r)) * random_01());

  PetscReal phi = 2.0 * M_PI * random_01();

  return Vector3R{
    ann.center[X] + r * cos(phi),
    ann.center[Y] + r * sin(phi),
    ann.center[Z] + ann.height * (random_01() - 0.5),
  };
}

Vector3R PreciseMomentum::operator()(const Vector3R& /* coordinate */)
{
  return value;
}

Vector3R thermal_momentum(const SortParameters& params)
{
  Vector3R T = {params.Tx, params.Ty, params.Tz};
  Vector3R result;

  for (Axis a : {X, Y, Z}) {
    PetscReal rand_vr = random_01();
    PetscReal rand_va = random_01();

    PetscReal sigma_p = sqrt(T[a] * params.m / mec2);
    result[a] = sigma_p * sqrt(-2.0 * log(rand_vr)) * sin(2.0 * M_PI * rand_va);
  }
  return result;
}

Vector3R thermal_velocity(const SortParameters& params)
{
  Vector3R velocity = thermal_momentum(params);
  velocity /= sqrt(params.m * params.m + velocity.squared());
  return velocity;
}

Vector3R MaxwellianMomentum::operator()(const Vector3R& /* coordinate */)
{
  Vector3R momentum = thermal_momentum(params);
  momentum[X] += params.px;
  momentum[Y] += params.py;
  momentum[Z] += params.pz;
  momentum /= sqrt(params.m * params.m + momentum.squared());
  return momentum;
}

Vector3R maxwellian_shifted_perturbation(  //
  const SortParameters& params, const BoxGeometry& box,
  const Vector3R& velocity, const Vector3R& wave_number, const Vector3R& phase,
  const Vector3R& coordinate, PetscReal (*trig_func)(PetscReal))
{
  PetscReal k[3]{
    2.0 * M_PI * wave_number[X] / (box.max[X] - box.min[X]),
    2.0 * M_PI * wave_number[Y] / (box.max[Y] - box.min[Y]),
    2.0 * M_PI * wave_number[Z] / (box.max[Z] - box.min[Z]),
  };

  Vector3R v_m = thermal_velocity(params);
  v_m[X] += velocity[X] * trig_func(k[X] * coordinate[X] + phase[X]);
  v_m[Y] += velocity[Y] * trig_func(k[Y] * coordinate[Y] + phase[Y]);
  v_m[Z] += velocity[Z] * trig_func(k[Z] * coordinate[Z] + phase[Z]);
  return v_m;
}

Vector3R MaxwellShiftedCosine::operator()(const Vector3R& coordinate)
{
  return maxwellian_shifted_perturbation(
    params, box, velocity, wave_number, phase, coordinate, cos);
}

Vector3R MaxwellShiftedSine::operator()(const Vector3R& coordinate)
{
  return maxwellian_shifted_perturbation(
    params, box, velocity, wave_number, phase, coordinate, sin);
}

KineticIonSoundMoments::KineticIonSoundMoments(const SortParameters& params,
  BoxGeometry box, PetscReal force_electric_amplitude, PetscReal omega_real,
  PetscReal gamma, const Vector3R& wave_number, const Vector3R& field_phase,
  const Vector3R& density_amplitude, const Vector3R& density_phase)
  : params(params),
    box(box),
    E_pa(force_electric_amplitude),
    E_pp(field_phase),
    n_pa(density_amplitude),
    n_pp(density_phase),
    omega_real(omega_real),
    gamma(gamma)
{
  PetscCheckAbort(gamma > 0 && isfinite(gamma) && isfinite(E_pa) && isfinite(omega_real), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "KineticIonSoundMoments requires finite plasma parameters: gamma > 0, E_pa, omega_real");

  PetscInt active = 0;

  for (Axis a : {X, Y, Z}) {
    PetscCheckAbort(isfinite(wave_number[a]) && isfinite(E_pp[a]) && isfinite(n_pa[a]) && isfinite(n_pp[a]), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "KineticIonSoundMoments requires finite mode parameters");

    if (wave_number[a] != 0) {
      axis = a;
      active++;
    }
    else {
      PetscCheckAbort(n_pa[a] == 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
        "KineticIonSoundMoments density perturbation must use the active wave axis");
    }
  }

  PetscCheckAbort(active == 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
    "KineticIonSoundMoments requires exactly one non-zero wave_number");

  PetscReal length = box.max[axis] - box.min[axis];
  PetscCheckAbort(length > 0 && isfinite(length), PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
    "KineticIonSoundMoments requires a finite non-empty box");

  PetscCheckAbort(abs(n_pa[axis]) < 1.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "KineticIonSoundMoments requires |density_amplitude| < 1");

  Vector3R T{params.Tx, params.Ty, params.Tz};
  PetscReal n0 = params.n;
  PetscReal q = params.q;
  PetscReal m = params.m;

  v2_0 = T[axis] / (m * mec2);
  PetscCheckAbort(v2_0 > 0 && isfinite(v2_0), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "KineticIonSoundMoments requires positive parallel temperature");

  PetscComplex imag_unit{0, 1.0};
  PetscComplex n_exp = 1.0 * exp(imag_unit * n_pp[axis]);
  PetscComplex E_exp = E_pa * exp(imag_unit * E_pp[axis]);

  PetscComplex omega{omega_real, -gamma};

  k_par = 2.0 * M_PI * wave_number[axis] / length;
  n_p = -imag_unit * n0 * n_pa[axis] * n_exp;
  v_p = omega * n_p / k_par;
  nv2_p = (omega * v_p - imag_unit * (q / m) * n0 * E_exp) / k_par;

  PetscReal n_min = n0 - abs(n_p);
  PetscReal v_max = /*  */ abs(v_p);
  PetscReal nv2_min = n0 * v2_0 - abs(nv2_p);
  PetscReal margin_determinant = n_min * nv2_min - v_max * v_max;
  PetscReal margin_subluminal = n_min - v_max;

  PetscCheckAbort(n_min > 0 && nv2_min > 0 && margin_determinant > 0 && margin_subluminal > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "KineticIonSoundMoments positivity check failed for '%s':                                                      \n"
    "n_min=%g (>0), P_min=%g (>0), n_min*P_min-|M1|^2=%g (>0), n_min-|M1|=%g (>0), |M1|=%g, |M2|=%g, k=%g.         \n"
    "Regenerate the mode constants for the configured plasma, box, and grid with ion_sound.py --theory --grid-dz.  \n",
    params.sort_name.c_str(), n_min, nv2_min, margin_determinant, margin_subluminal, v_max, abs(nv2_p), k_par);
}

void KineticIonSoundMoments::local_parallel_parameters(const Vector3R& coordinate,
  PetscReal& bulk_velocity, PetscReal& parallel_scale) const
{
  PetscReal phase = k_par * (coordinate[axis] - box.min[axis]);
  PetscComplex exp = std::polar(1.0, phase);

  // local values of perturbed quantities, according to ion-sound profiles
  PetscReal n0 = params.n;
  PetscReal n = n0 + std::real(n_p * exp);
  PetscReal v = /*  */ std::real(v_p * exp);
  PetscReal nv2 = n0 * v2_0 + std::real(nv2_p * exp);

  bulk_velocity = v / n;
  PetscReal v2 = nv2 / n - bulk_velocity * bulk_velocity;

  PetscCheckAbort(n > 0 && v2 > 0 && abs(bulk_velocity) < 1.0 && isfinite(v2), PETSC_COMM_WORLD, PETSC_ERR_ARG_CORRUPT,
    "KineticIonSoundMoments encountered invalid local moments");

  parallel_scale = sqrt(v2 / v2_0);
}

Vector3R thermal_velocity_quiet(const SortParameters& params, std::size_t index)
{
  Vector3R T = {params.Tx, params.Ty, params.Tz};
  Vector3R result;

  for (Axis a : {X, Y, Z}) {
    PetscReal rand_vr = van_der_corput(index, velocity_radius_bases[a]);
    PetscReal rand_va = van_der_corput(index, velocity_phase_bases[a]);

    PetscReal sigma_v = sqrt(T[a] / (params.m * mec2));
    result[a] = sigma_v * sqrt(-2.0 * log(rand_vr)) * sin(2.0 * M_PI * rand_va);
  }
  return result;
}

Vector3R KineticIonSoundMoments::operator()(const Vector3R& coordinate)
{
  if (return_second_velocity) {
    PetscCheckAbort((coordinate - first_coordinate).length() == 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
      "KineticIonSoundMoments requires consecutive duplicate coordinates");

    return_second_velocity = false;
    return second_velocity;
  }

  PetscReal bulk_velocity = 0;
  PetscReal parallel_scale = 0;
  local_parallel_parameters(coordinate, bulk_velocity, parallel_scale);

  PetscCheckAbort(pair_number < std::numeric_limits<std::size_t>::max(), PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
    "KineticIonSoundMoments exhausted its quiet sequence");

  Vector3R thermal_velocity = thermal_velocity_quiet(params, pair_number++);
  thermal_velocity[axis] *= parallel_scale;

  Vector3R first_velocity = thermal_velocity;
  first_velocity[axis] += bulk_velocity;

  second_velocity = -1.0 * thermal_velocity;
  second_velocity[axis] += bulk_velocity;

  /// @note If `pair_number` starts with 1, my guess is that it would properly generate velocities
  PetscCheckAbort(first_velocity.length() < 1.0 && isfinite(first_velocity.abs_max()), PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED,
    "KineticIonSoundMoments could not sample first velocity in a subluminal pair");

  PetscCheckAbort(second_velocity.length() < 1.0 && isfinite(second_velocity.abs_max()), PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED,
    "KineticIonSoundMoments could not sample second velocity in a subluminal pair");

  first_coordinate = coordinate;
  return_second_velocity = true;
  return first_velocity;
}

Vector3R AngularMomentum::operator()(const Vector3R& coordinate)
{
  PetscReal x = coordinate[X] - center[X];
  PetscReal y = coordinate[Y] - center[Y];
  PetscReal r = hypot(x, y);

  Vector3R momentum = thermal_momentum(params);

  if (isinf(1.0 / r))
    return Vector3R{0, 0, params.pz} + momentum;

  return Vector3R{
    -params.px * (y / r) + momentum[X],
    +params.py * (x / r) + momentum[Y],
    +params.pz + momentum[Z],
  };
}
