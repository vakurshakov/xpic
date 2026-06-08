#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic pusher: (1) uniform ExB drift, (2) acceleration along\n"
  "E in constant magnetic field. Particle guiding center should drift with\n"
  "velocity (ExB/B^2).\n";

constexpr Vector3R E0(0, 0.005, -0.1);
constexpr Vector3R B0(0, 0, 0.2);

void get_ExB_field(
  const Vector3R&, const Vector3R&, Vector3R& E_p, Vector3R& B_p, Vector3R&)
{
  E_p = E0;
  B_p = B0;
}

void get_B(
  const Vector3R&, const Vector3R&, Vector3R& B0_p, Vector3R& meanB_p, Vector3R& Bn_p)
{
  B0_p = B0;
  meanB_p = B0;
  Bn_p = B0;
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt = 100;

  dt = omega_dt / B0.length();
  geom_nt = 1;
  diagnose_period = 1;

  constexpr Vector3R r0(0, 0, 0);
  constexpr Vector3R v0(0.05, 0, 0);
  Point point_boris(r0, v0);
  PointByField point_dk(point_boris, B0, 1, q / m);

  PointByFieldTrace trace_dk(__FILE__, std::format("drift_kinetic_100"), point_dk);
  PointTrace trace_boris(__FILE__, std::format("boris_100"), point_boris);

  DriftKineticPush dk_push;
  dk_push.set_qm(q / m);
  dk_push.set_mp(m);
  dk_push.set_fields_callback(get_ExB_field);
  dk_push.set_B_callback(get_B);

  BorisPush boris_push;
  boris_push.set_qm(q / m);

  Vector3R start_r = point_dk.r;
  const PetscReal old_E =
    point_dk.p_parallel * point_dk.p_parallel + point_dk.p_perp * point_dk.p_perp;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_0 = point_dk;

    PetscCall(trace_dk.diagnose(t));
    dk_push.process(dt, point_dk, point_0);

    PetscCall(trace_boris.diagnose(t));

    boris_step(boris_push, point_boris, get_ExB_field);
  }

  PetscReal T = dt * (PetscReal)(geom_nt + 1);
  PetscReal q = dk_push.get_qm() * dk_push.get_mp();
  PetscReal E_par = E0.z();
  PetscReal p_par_theory = q * E_par * T;

  PetscCheck(equal_tol(point_dk.p_parallel, p_par_theory, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "p_parallel should be q*E*t. Result: %.6e, theory: %.6e", point_dk.p_parallel, p_par_theory);

  PetscReal z_theory = 0.5 * q * E_par * T * T;

  PetscCheck(equal_tol(point_dk.z(), z_theory, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "z should be 0.5*q*E*t^2. Result: %.6e, theory: %.6e", point_dk.z(), z_theory);

  Vector3R V_drift = E0.cross(B0) / (B0.length() * B0.length());
  Vector3R r_theory = start_r + V_drift * T + Vector3R{0, 0, z_theory};

  PetscCheck(equal_tol(point_dk.r, r_theory, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Guiding center must move with ExB drift. Result: (%.6e %.6e %.6e), theory: (%.6e %.6e %.6e)", REP3_A(point_dk.r), REP3_A(r_theory));

  const PetscReal new_E =
    point_dk.p_parallel * point_dk.p_parallel + point_dk.p_perp * point_dk.p_perp;
  const PetscReal theory_E = old_E + p_par_theory * p_par_theory;

  PetscCheck(equal_tol(new_E, theory_E, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Energy must change only due to work of E_parallel. Result: %.6e, theory: %.6e", new_E, theory_E);

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
