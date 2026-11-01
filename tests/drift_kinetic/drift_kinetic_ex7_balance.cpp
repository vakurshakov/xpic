#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Test of the invariants for \"drift_kinetic\" implementation with evolving\n"
  "fields. Maxwellian electrons (ions are stationary background) with T=0.1\n"
  "gyrate in B=1 along z and are advanced for 2 cycles in periodic         \n"
  "boundaries. The change of the field energy must be compensated by the   \n"
  "kinetic one, dE + dB + dK = 0, and the continuity equation must hold,   \n"
  "div(J) + d(rho)/dt = 0. Particle count and mu of every particle are     \n"
  "conserved, the fields stay finite and bounded.                          \n";

static constexpr PetscReal temperature = 0.1;
static constexpr PetscReal B0 = 1.0;

void overwrite_config();

struct Moments {
  PetscInt count = 0;
  PetscReal mu_sum = 0;
  PetscReal mu_sq = 0;
  PetscReal mu_min = PETSC_MAX_REAL;
  PetscReal mu_max = 0;
  PetscReal p_par = 0;
};

static PetscErrorCode collect_moments(
  const drift_kinetic::Particles& particles, Moments* m);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();
  Configuration::save(get_out_dir(__FILE__));

  drift_kinetic::Simulation simulation;
  PetscCall(simulation.initialize());

  Moments m0, m1;
  PetscCall(collect_moments(*simulation.particles_[0], &m0));
  PetscCall(simulation.calculate());
  PetscCall(collect_moments(*simulation.particles_[0], &m1));

  PetscReal E_max, B_max;
  PetscCall(VecNorm(simulation.E, NORM_INFINITY, &E_max));
  PetscCall(VecNorm(simulation.B, NORM_INFINITY, &B_max));

  PetscCall(simulation.finalize());

  PetscReal max_dK, max_err, max_dQ;
  PetscCall(max_abs_column(__FILE__, "dk_diagnostic.txt", "dK", max_dK));
  PetscCall(max_abs_column(__FILE__, "dk_diagnostic.txt", "dE+dB+dK", max_err));
  PetscCall(max_abs_column(__FILE__, "charge_conservation.txt", "N2dQ_tot", max_dQ));

  PetscCheck(max_dK > 1e-6, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Kinetic energy change is too small to be checked: %g", max_dK);

  PetscCheck(max_err < 1e-4 * max_dK, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Energy balance is violated, max error: %g, max dK: %g", max_err, max_dK);

  PetscCheck(max_dQ < 1e-10, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Charge is not conserved, max error: %g", max_dQ);

  PetscCheck(m1.count == m0.count, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Number of particles has changed: %d -> %d", m0.count, m1.count);

  // The same `mu` values are carried by the particles, up to the summation order.
  bool mu_conserved =                                  //
    equal_rel(m1.mu_sum, m0.mu_sum, 1e-12, 0) &&       //
    equal_rel(m1.mu_sq, m0.mu_sq, 1e-12, 0) &&         //
    m1.mu_min == m0.mu_min && m1.mu_max == m0.mu_max;  //

  PetscCheck(mu_conserved, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Magnetic moment is not conserved: sum %.15e -> %.15e, max %.15e -> %.15e", m0.mu_sum, m1.mu_sum, m0.mu_max, m1.mu_max);

  // The electrons have no neutralizing background, so the total momentum is
  // not an exact invariant, only its drift is bounded by the thermal scale.
  const PetscReal p_scale = m0.count * std::sqrt(temperature);
  PetscCheck(std::abs(m1.p_par - m0.p_par) < 0.1 * p_scale, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Total parallel momentum has drifted: %g -> %g", m0.p_par, m1.p_par);

  PetscCheck(std::isfinite(E_max) && E_max < 0.1, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Electric field is not bounded: %g", E_max);

  PetscCheck(std::isfinite(B_max) && std::abs(B_max - B0) < 0.01, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Magnetic field is not bounded: %g", B_max);

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config()
{
  dx = 0.5;
  geom_nx = 8;
  geom_x = geom_nx * dx;

  dt = 1.0;
  geom_nt = 2;
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", "drift_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {
      "Geometry",
      {
        {"x", geom_x},
        {"y", geom_x},
        {"z", geom_x},
        {"t", geom_t},
        {"dx", dx},
        {"dy", dx},
        {"dz", dx},
        {"dt", dt},
        {"diagnose_period", dt},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "electrons"},
        {"Np", 1},
        {"n", +1.0},
        {"q", -1.0},
        {"m", +1.0},
        {"T", temperature},
      }},
    },
    {
      "Presets",
      {
        {
          {"command", "SetField"},
          {"field", "B"},
          {"setter", {{"name", "SetUniformField"}, {"value", {0.0, 0.0, B0}}}},
        },
        {
          {"command", "SetParticles"},
          {"particles", "electrons"},
          {"coordinate", {{"name", "CoordinateInBox"}}},
          {"momentum", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
        },
      },
    },
  });
}

static PetscErrorCode collect_moments(
  const drift_kinetic::Particles& particles, Moments* m)
{
  PetscFunctionBeginUser;
  ParticleStorage::Attributes f;
  PetscCall(particles.storage.get_attributes(f));
  const drift_kinetic::DkPointArrays arrays(f);

  for (PetscInt i = 0; i < f.size; i++) {
    const PointByField point = arrays.load(i);
    m->count++;
    m->mu_sum += point.mu();
    m->mu_sq += POW2(point.mu());
    m->mu_min = std::min(m->mu_min, point.mu());
    m->mu_max = std::max(m->mu_max, point.mu());
    m->p_par += point.p_par();
  }
  PetscCall(particles.storage.restore_attributes(f));

  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(MPI_IN_PLACE, &m->count, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(MPI_IN_PLACE, &m->mu_sum, 2, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(MPI_IN_PLACE, &m->mu_min, 1, MPIU_REAL, MPI_MIN, PETSC_COMM_WORLD));
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(MPI_IN_PLACE, &m->mu_max, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD));
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(MPI_IN_PLACE, &m->p_par, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}
