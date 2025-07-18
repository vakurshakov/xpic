// drift_kinetic_push_ex_ExB.cpp
#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic pusher: uniform ExB drift.\n"
  "Particle guiding center should drift with velocity E x B / B^2.\n";

constexpr Vector3R E0(0.0, 0.0, 1.0);
constexpr Vector3R B0(0.0, 0.0, 1.0);

void get_ExB_field(const Vector3R&, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = E0;
  B_p = B0;
  gradB_p = {0.0, 0.0, 0.0};
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  constexpr Vector3R r0(0.0, 0.0, 0.0);
  constexpr Vector3R v0(0.0, 1.0, 0.0);
  Point point_init(r0, v0);

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  std::string id = std::format("omega_dt_{:.1f}", omega_dt);

  dt = omega_dt / B0.length();
  geom_nt = 100'000;
  diagnose_period = geom_nt / 4;

  PointByField point_n(point_init, B0, 1.0);

  PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 123);


  DriftKineticPush push;
  push.set_qm(-1.0);
  push.set_mp(1.0);
  push.set_fields_callback(get_ExB_field);

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_0 = point_n;

    PetscCall(trace.diagnose(t));
    push.process(dt, point_n, point_0);
  }

  PetscReal T = dt * (geom_nt + 1.);
  PetscReal q = push.get_qm();
  PetscReal E_par = E0.z();
  PetscReal p_par_theory = q * E_par * T;

  PetscCheck(equal_tol(point_n.p_parallel, p_par_theory, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "p_parallel should be q*E*t. Result: %.6e, theory: %.6e",
    point_n.p_parallel, p_par_theory);

  PetscReal z_theory = 0.5 * q * E_par * T * T;

  PetscCheck(equal_tol(point_n.r.z(), z_theory, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "z should be 0.5*q*E*t^2. Result: %.6e, theory: %.6e",
    point_n.r.z(), z_theory);

  PetscFinalize();
  return EXIT_SUCCESS;
}
