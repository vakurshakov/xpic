#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test: Magnetic mirror field with realistic analytic mirrors.\n"
  "Particle should remain trapped in the well; energy conserved.\n";

using namespace drift_kinetic_test_utils;
using namespace gaussian_magnetic_mirror;

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  constexpr PetscReal v_perp = 0.2;
  constexpr PetscReal v_par = 0.3;
  constexpr Vector3R r0(Rc + 0.5, Rc, L);
  constexpr Vector3R v0(v_perp, 0, v_par);
  Point point_init(r0, v0);
  PointByField point_n(point_init, {0, 0, get_Bz_corr(r0)}, 1, q / m);

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dt = omega_dt / get_Bz(r0.z() - L);
  geom_nt = 300'000;
  diagnose_period = geom_nt / 2;

  auto id = std::format("omega_dt_{:.1f}", omega_dt);
  PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 123);

  DriftKineticPush push;
  push.set_qm(q / m);
  push.set_mp(m);
  push.set_fields_callback(get_fields);

  PetscReal z_max = L;
  PetscReal r_max = Rc * 1.5;

  const PetscReal old_E = get_kinetic_energy(point_n);

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_0 = point_n;
    push.process(dt, point_n, point_0);

    PetscCall(trace.diagnose(t));

    PetscCheck(std::abs(point_n.z() - L) <= z_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle escaped mirror! z = %.6e, allowed = %.6e", point_n.z() - L, z_max);

    PetscReal r = std::hypot(point_n.x() - Rc, point_n.y() - Rc);

    PetscCheck(r <= r_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle escaped radial well! r = %.6e, allowed = %.6e", r, r_max);
  }

  const PetscReal new_E = get_kinetic_energy(point_n);

  PetscCheck(equal_tol(new_E, old_E, 1e-6), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Energy not conserved: new = %.6e, old = %.6e", new_E, old_E);

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
