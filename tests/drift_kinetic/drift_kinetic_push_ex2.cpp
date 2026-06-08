#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic pusher: gradient drift.\n"
  "Magnetic field B = B0 * (1 + alpha*y) * z (linear gradient along y).\n"
  "Particle starts with pure perpendicular velocity -> p_parallel = 0.\n"
  "Gradient drift: V_grad = (mu/q) * (b x gradB) / B -> drift along x.\n";

constexpr PetscReal alpha = -0.8;
constexpr Vector3R B0_dir(0, 0, 1);
constexpr PetscReal B0_mag = 0.2;

// Linear field: Bz(y) = B0 * (1 + alpha*y) = 0.2 * (1 - 0.8*y)
PetscReal B_mag(const Vector3R& r)
{
  return B0_mag * (1.0 + alpha * r.y());
}

// B = B0*(1 + alpha*y) => gradB = alpha*B0 * y_hat (constant)
Vector3R gradB(const Vector3R&)
{
  return Vector3R(0, alpha * B0_mag, 0);
}

void get_fields_dk(
  const Vector3R& r0, const Vector3R& rn, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  Vector3R rmid = 0.5 * (r0 + rn);
  E_p = {};
  B_p = B_mag(rmid) * B0_dir;
  gradB_p = gradB(rmid);
}

void get_fields_boris(
  const Vector3R& r0, const Vector3R&, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = {};
  B_p = B_mag(r0) * B0_dir;
  gradB_p = gradB(r0);
}

void get_B(
  const Vector3R& r0, const Vector3R& rn, Vector3R& B0_p, Vector3R& meanB_p, Vector3R& Bn_p)
{
  B0_p = B_mag(r0) * B0_dir;
  meanB_p = B_mag(0.5 * (r0 + rn)) * B0_dir;
  Bn_p = B_mag(rn) * B0_dir;
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt = 100;
  dt = omega_dt / B0_mag;
  geom_nt = 1;
  diagnose_period = 1;

  constexpr Vector3R r0(0, 0, 0);
  // pure perpendicular velocity -> p_parallel = 0
  constexpr Vector3R v0(0, 0.05, 0);
  Point point_init(r0, v0);

  Vector3R B0_at_r0 = B_mag(r0) * B0_dir;
  PointByField dk_point(point_init, B0_at_r0, 1, q / m);
  Point boris_point(point_init);

  const PetscReal mu_init = dk_point.mu_p;
  // DK energy invariant: p_parallel^2 + 2*mu*B(guiding_center).
  // Cannot use raw p_perp here: the PointByField ctor fills p_perp from the
  // Larmor velocity at r0 (|B|=1), while the pusher later renormalizes it to
  // sqrt(2*mu*B) at the guiding center (y=1, |B|=1/1.01).
  const PetscReal old_energy =
    dk_point.p_parallel * dk_point.p_parallel + 2.0 * mu_init * B_mag(dk_point.r);

  // v_∇B = (mu/q) * (b x gradB) / B, evaluated at the guiding center.
  // b = z, gradB_y = alpha*B0 (constant), B = B0*(1 + alpha*y)
  //   => v_∇B_x = -(mu/q) * alpha / (1 + alpha*y_gc)
  // y_gc не меняется (нет дрейфа по y) -> постоянная скорость, x(t) линейна.
  PetscReal V_grad_x =
    -(mu_init / (q / m * m)) * alpha / (1.0 + alpha * dk_point.r.y());

  PointByFieldTrace dk_trace(__FILE__, std::format("drift_kinetic_100"), dk_point);
  PointTrace boris_trace(__FILE__, std::format("boris_100"), boris_point);

  DriftKineticPush dk_push;
  dk_push.set_qm(q / m);
  dk_push.set_mp(m);
  dk_push.set_fields_callback(get_fields_dk);
  dk_push.set_B_callback(get_B);

  BorisPush boris_push;
  boris_push.set_qm(q / m);

  const Vector3R start_r = dk_point.r;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField dk_point_old = dk_point;

    PetscCall(dk_trace.diagnose(t));
    PetscCall(boris_trace.diagnose(t));

    dk_push.process(dt, dk_point, dk_point_old);
    boris_step(boris_push, boris_point, get_fields_boris);
  }

  const PetscReal total_time = dt * (PetscReal)(geom_nt + 1);
  const Vector3R r_theory = start_r + Vector3R(V_grad_x * total_time, 0, 0);

  PetscCheck(equal_tol(dk_point.r, r_theory, 1e-2), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Guiding center must drift with grad-B drift. Result: (%.6e %.6e %.6e), theory: (%.6e %.6e %.6e)",
    REP3_A(dk_point.r), REP3_A(r_theory));

  const PetscReal new_energy =
    dk_point.p_parallel * dk_point.p_parallel + 2.0 * dk_point.mu_p * B_mag(dk_point.r);

  PetscCheck(equal_tol(new_energy, old_energy, 1e-6), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Energy must be conserved without electric field. Result: %.6e, initial: %.6e",
    new_energy, old_energy);

  PetscCheck(equal_tol(dk_point.mu_p, mu_init, 1e-10), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Magnetic moment must be conserved. Result: %.6e, initial: %.6e",
    dk_point.mu_p, mu_init);

  PetscCheck(equal_tol(dk_point.p_parallel, 0.0, 1e-6), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "p_parallel must stay zero in pure grad-B configuration. Result: %.6e",
    dk_point.p_parallel);

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
