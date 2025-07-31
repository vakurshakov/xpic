#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic pusher: grad-B drift in non-uniform B field.\n"
  "Guiding center should drift with classical grad-B velocity.\n";

constexpr Vector3R B0(0.0, 0.0, 2.0);
constexpr Vector3R gradB0(1.0, 0.0, 0.0);

void get_gradB_field(
  const Vector3R& r, Vector3R&, Vector3R& B_p, Vector3R& gradB_p)
{
  B_p = B0 + r.elementwise_product(gradB0);
  gradB_p = gradB0;
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dt = omega_dt / B0.length();
  geom_nt = 100'000;
  diagnose_period = geom_nt / 4;

  constexpr Vector3R r0(0.0, 0.0, 0.0);
  constexpr Vector3R v0(1.0, 0.0, 1.0);
  Point point_init(r0, v0);
  PointByField point_n(point_init, B0, 1.0);

  auto id = std::format("omega_dt_{:.1f}", omega_dt);
  PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 123);

  DriftKineticPush push;
  push.set_qm(1.0);
  push.set_mp(1.0);
  push.set_fields_callback(get_gradB_field);

  Vector3R start_r = point_n.r;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_0 = point_n;

    PetscCall(trace.diagnose(t));
    push.process(dt, point_n, point_0);
  }

  PetscReal m = push.get_mp();
  PetscReal q = push.get_qm();
  PetscReal B = B0.length();

  Vector3R V_gradB;
  V_gradB += Vector3R(0.0, 0.0, v0.z());
  V_gradB += POW2(v0.z()) / POW3(B) * B0.cross(gradB0);
  V_gradB += point_n.mu_p / (m * q * POW2(B)) * B0.cross(gradB0);

  PetscReal T = dt * (PetscReal)(geom_nt + 1);
  Vector3R r_theory = start_r + V_gradB * T;

  PetscCheck(equal_tol(point_n.r, r_theory, 1e-8), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Guiding center must drift with grad-B drift. Result: (%.8e %.8e %.8e), theory: (%.8e %.8e %.8e)", REP3_A(point_n.r), REP3_A(r_theory));

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
