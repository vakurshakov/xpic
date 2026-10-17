#include "set_particles_iaw.h"

#include "src/diagnostics/energy.h"

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

}  // namespace

SetParticlesIonAcousticWave::SetParticlesIonAcousticWave(
  interfaces::Particles& particles, PetscInt number_of_pairs,
  PetscReal force_electric_amplitude, PetscReal omega_real, PetscReal gamma,
  const Vector3R& wave_number, const Vector3R& field_phase,
  const Vector3R& density_amplitude, const Vector3R& density_phase)
  : particles(particles), number_of_pairs(number_of_pairs)
{
  PetscCheckAbort(gamma > 0 && isfinite(gamma) && isfinite(force_electric_amplitude) && isfinite(omega_real), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "SetParticlesIonAcousticWave requires finite plasma parameters: gamma > 0, force_electric_amplitude, omega_real");

  PetscInt active = 0;

  for (Axis a : {X, Y, Z}) {
    PetscCheckAbort(isfinite(wave_number[a]) && isfinite(field_phase[a]) && isfinite(density_amplitude[a]) && isfinite(density_phase[a]), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "SetParticlesIonAcousticWave requires finite mode parameters");

    if (wave_number[a] != 0) {
      axis = a;
      active++;
    }
    else {
      PetscCheckAbort(density_amplitude[a] == 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
        "SetParticlesIonAcousticWave density perturbation must use the active wave axis");
    }
  }

  PetscCheckAbort(active == 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
    "SetParticlesIonAcousticWave requires exactly one non-zero wave_number");

  n_pa = density_amplitude[axis];
  n_pm = wave_number[axis];
  n_pp = density_phase[axis];

  PetscCheckAbort(abs(n_pa) < 1.0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "SetParticlesIonAcousticWave requires |density_amplitude| < 1");

  PetscReal round_pm = round(n_pm);
  PetscCheckAbort(round_pm > 0 && abs(n_pm - round_pm) <= 1e-12, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "SetParticlesIonAcousticWave requires a positive integer wave_number");

  const SortParameters& params = particles.parameters;
  length = Vector3R{geom_x, geom_y, geom_z}[axis];

  Vector3R T{params.Tx, params.Ty, params.Tz};
  PetscReal n0 = params.n;
  PetscReal q = params.q;
  PetscReal m = params.m;

  v2_0 = T[axis] / (m * mec2);
  PetscCheckAbort(v2_0 > 0 && isfinite(v2_0), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "SetParticlesIonAcousticWave requires positive parallel temperature");

  PetscReal E_pa = force_electric_amplitude;
  PetscReal E_pp = field_phase[axis];

  PetscComplex imag_unit{0, 1.0};
  PetscComplex n_exp = 1.0 * exp(imag_unit * n_pp);
  PetscComplex E_exp = E_pa * exp(imag_unit * E_pp);

  PetscComplex omega{omega_real, -gamma};

  k_par = 2.0 * M_PI * n_pm / length;
  n_p = -imag_unit * n0 * n_pa * n_exp;
  v_p = omega * n_p / k_par;
  nv2_p = (omega * v_p - imag_unit * (q / m) * n0 * E_exp) / k_par;

  PetscReal n_min = n0 - abs(n_p);
  PetscReal v_max = /*  */ abs(v_p);
  PetscReal nv2_min = n0 * v2_0 - abs(nv2_p);
  PetscReal margin_determinant = n_min * nv2_min - v_max * v_max;
  PetscReal margin_subluminal = n_min - v_max;

  PetscCheckAbort(n_min > 0 && nv2_min > 0 && margin_determinant > 0 && margin_subluminal > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "SetParticlesIonAcousticWave positivity check failed for '%s':                                                 \n"
    "n_min=%g (>0), P_min=%g (>0), n_min*P_min-|M1|^2=%g (>0), n_min-|M1|=%g (>0), |M1|=%g, |M2|=%g, k=%g.         \n"
    "Regenerate the mode constants for the configured plasma, box, and grid with ion_sound.py --theory --grid-dz.  \n",
    params.sort_name.c_str(), n_min, nv2_min, margin_determinant, margin_subluminal, v_max, abs(nv2_p), k_par);
}

PetscErrorCode SetParticlesIonAcousticWave::execute(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  added_energy = 0;
  added_particles = 0;

  const SortParameters& params = particles.parameters;
  const PetscReal mpw = params.n / params.Np;

  for (PetscInt p = 0; p < number_of_pairs; p++) {
    Vector3R coordinate = pair_coordinate(p);

    PetscReal bulk_velocity = 0;
    PetscReal parallel_scale = 0;
    local_parallel_parameters(coordinate, bulk_velocity, parallel_scale);

    Vector3R thermal_velocity = thermal_velocity_quiet(params, p + 1);
    thermal_velocity[axis] *= parallel_scale;

    for (PetscReal sign : {+1.0, -1.0}) {
      Vector3R velocity = sign * thermal_velocity;
      velocity[axis] += bulk_velocity;

      PetscCheck(velocity.length() < 1.0 && isfinite(velocity.abs_max()), PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED,
        "SetParticlesIonAcousticWave could not sample a subluminal velocity");

      bool is_added = false;
      PetscCall(particles.add_particle(Point(coordinate, velocity), &is_added));

      if (is_added) {
        added_energy += Energy::get_kinetic(velocity, params.m, mpw);
        added_particles++;
      }
    }
  }

  PetscCall(particles.commit_added_particles());
  PetscCall(log_statistics());
  PetscFunctionReturn(PETSC_SUCCESS);
}

Vector3R SetParticlesIonAcousticWave::pair_coordinate(PetscInt p) const
{
  Vector3R coordinate;

  for (Axis a : {X, Y, Z}) {
    PetscReal rand_x = (a == axis)
      ? sine_density_inv_cdf((p + 0.5) / number_of_pairs, n_pa, n_pm, n_pp)
      : van_der_corput(p + 1, coordinate_bases[a]);

    coordinate[a] = rand_x * Geom[a];
  }
  return coordinate;
}

void SetParticlesIonAcousticWave::local_parallel_parameters(
  const Vector3R& coordinate, PetscReal& bulk_velocity,
  PetscReal& parallel_scale) const
{
  PetscComplex exp = std::polar(1.0, k_par * coordinate[axis]);

  // local values of perturbed quantities, according to ion-sound profiles
  PetscReal n0 = particles.parameters.n;
  PetscReal n = n0 + std::real(n_p * exp);
  PetscReal v = /*  */ std::real(v_p * exp);
  PetscReal nv2 = n0 * v2_0 + std::real(nv2_p * exp);

  bulk_velocity = v / n;
  PetscReal v2 = nv2 / n - bulk_velocity * bulk_velocity;

  PetscCheckAbort(n > 0 && v2 > 0 && abs(bulk_velocity) < 1.0 && isfinite(v2), PETSC_COMM_WORLD, PETSC_ERR_ARG_CORRUPT,
    "SetParticlesIonAcousticWave encountered invalid local moments");

  parallel_scale = sqrt(v2 / v2_0);
}

PetscErrorCode SetParticlesIonAcousticWave::log_statistics()
{
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &added_energy, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &added_particles, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

  LOG("  Particles have been added into \"{}\"", particles.parameters.sort_name);
  PetscCall(MPIUtils::log_statistics("    ", added_particles, PETSC_COMM_WORLD));
  LOG("    energy: {:6.4e}", added_energy);
  PetscCall(particles.log_distribution());
  PetscFunctionReturn(PETSC_SUCCESS);
}
