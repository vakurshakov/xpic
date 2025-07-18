// drift_kinetic_push_ex_ExB.cpp
#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic pusher: uniform ExB drift.\n"
  "Particle guiding center should drift with velocity E x B / B^2.\n";

constexpr Vector3R E0(0.0, 1.0, 0.0);
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

  Vector3R start_r = point_n.r;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_0 = point_n;

    PetscCall(trace.diagnose(t));
    push.process(dt, point_n, point_0);
  }

  Vector3R Vdrift = E0.cross(B0) / (B0.length() * B0.length());
  std::cout << Vdrift.x() << " " << Vdrift.y() << " " << Vdrift.z() << std::endl;
  PetscReal T = dt * (geom_nt+1);
  Vector3R r_theory = start_r + (Vdrift) * T;

  PetscCheck(equal_tol(point_n.r, r_theory, 1e-5), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Guiding center must move with ExB drift. Result: (%.6e %.6e %.6e), theory: (%.6e %.6e %.6e)",
    point_n.r.x(), point_n.r.y(), point_n.r.z(),
    r_theory.x(), r_theory.y(), r_theory.z());

  PetscFinalize();
  return EXIT_SUCCESS;
}
