#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test: Magnetic mirror field with realistic analytic mirrors.\n"
  "Particle should remain trapped in the well; energy conserved.\n";

int main(int argc, char** argv)
{
  using namespace gaussian_magnetic_mirror;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  constexpr PetscReal v_perp = 1.0;
  constexpr PetscReal v_par = 0.6;
  constexpr Vector3R r0(0.5, 0.0, 0.0);
  constexpr Vector3R v0(v_perp, 0.0, v_par);
  Point point_init(r0+correction::rho(v0, Vector3R(0.0, 0.0, get_Bz_corr(r0)), q/m), v0);
  PointByField point_n(point_init, {0.0, 0.0, get_Bz_corr(r0)}, 1.0, q/m);

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dt = omega_dt / get_Bz(r0.z());
  geom_nt = 300'000;
  diagnose_period = geom_nt / 2;

  auto id = std::format("omega_dt_{:.1f}", omega_dt);
  PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 123);

  DriftKineticPush push;
  push.set_qm(q/m);
  push.set_mp(m);
  push.set_fields_callback(get_fields);

  PetscReal z_max = L + 1e-2;
  PetscReal r_max = L + 1e-2;

  const PetscReal old_E = //
    point_n.p_parallel * point_n.p_parallel + //
    point_n.p_perp * point_n.p_perp;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_0 = point_n;
    push.process(dt, point_n, point_0);

    PetscCall(trace.diagnose(t));

    PetscCheck(std::abs(point_n.r.z()) <= z_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle escaped mirror! z = %.6e, allowed = %.6e", point_n.r.z(), z_max);

    PetscCheck(point_n.r.length() <= r_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle escaped radial well! r = %.6e, allowed = %.6e", point_n.r.length(), r_max);
  }

  const PetscReal new_E = //
    point_n.p_parallel * point_n.p_parallel + //
    point_n.p_perp * point_n.p_perp;

  PetscCheck(equal_tol(new_E, old_E, 1e-6), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Energy not conserved: new = %.6e, old = %.6e", new_E, old_E);

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
