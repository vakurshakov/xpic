#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic pusher: larmor center (guiding center) should\n"
  "remain fixed and energy should be conserved in a constant B field.\n";

constexpr Vector3R B0(0.0, 0.0, 2.0);

void get_magnetic_field(const Vector3R&, const Vector3R&, Vector3R&, Vector3R& B_p, Vector3R&)
{
  B_p = B0;
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dt = omega_dt / B0.length();
  geom_nt = 1'000;
  diagnose_period = geom_nt / 4;

  constexpr Vector3R r0(0.0, 0.0, 0.0);
  constexpr Vector3R v0(0.0, 1.0, 0.0);
  Point point_init(r0, v0);
  PointByField point_n(point_init, B0, 1.0);

  auto id = std::format("omega_dt_{:.1f}", omega_dt);
  PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 123);

  DriftKineticPush push;
  push.set_qm(-1.0);
  push.set_mp(1.0);
  push.set_fields_callback(get_magnetic_field);

  const PetscReal old_E = //
    point_n.p_parallel * point_n.p_parallel + //
    point_n.p_perp * point_n.p_perp;

  Vector3R start_r = point_n.r;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_0 = point_n;

    PetscCall(trace.diagnose(t));
    push.process(dt, point_n, point_0);
  }

  PetscCheck(equal_tol(point_n.r, start_r, 1e-10), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Larmor center must remain fixed in uniform B (result: %.10e %.10e %.10e, expected: %.10e %.10e %.10e)", REP3_A(point_n.r), REP3_A(start_r));

  const PetscReal new_E = //
    point_n.p_parallel * point_n.p_parallel + //
    point_n.p_perp * point_n.p_perp;

  PetscCheck(equal_tol(new_E, old_E, PETSC_SMALL), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Energy must be conserved in uniform B. Result: %.10e, initial: %.10e", new_E, old_E);

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}

