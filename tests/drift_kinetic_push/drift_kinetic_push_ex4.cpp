#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test: Magnetic mirror field with radial well.\n"
  "Particle should remain trapped in the well; energy conserved.\n";

using namespace quadratic_magnetic_mirror;

constexpr PetscReal phi = 0.5;
constexpr PetscReal E_phi = 0.3;

void get_fields(const Vector3R&, const Vector3R& rn, Vector3R& E_p,
  Vector3R& B_p, Vector3R& gradB_p)
{
  quadratic_magnetic_mirror::get_fields(rn, rn, E_p, B_p, gradB_p);

  E_p = Vector3R{
    +E_phi * (rn.y() - Rc),
    -E_phi * (rn.x() - Rc),
    +phi * M_PI / D * std::sin(M_PI * (rn.z() - L) / D),
  };
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  constexpr PetscReal v_perp = 0.1;
  constexpr PetscReal v_par = 0.1;
  constexpr Vector3R r0(Rc + 0.5, Rc, L);
  constexpr Vector3R v0(v_perp, 0.0, v_par);

  PetscReal B = get_B(hypot(r0.x() - Rc, r0.y() - Rc), r0.z() - L);

  Point point_init(r0, v0);
  PointByField point_n(point_init, {0.0, 0.0, B}, 1.0, q / m);

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dt = omega_dt / B;
  geom_nt = 10'000;
  diagnose_period = geom_nt / 4;

  auto id = std::format("omega_dt_{:.1f}", omega_dt);
  PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 1000);

  DriftKineticPush push;
  push.set_qm(q / m);
  push.set_mp(m);
  push.set_fields_callback(::get_fields);

  PetscReal z_max = L;
  PetscReal r_max = Rc * 1.5;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_0 = point_n;
    push.process(dt, point_n, point_0);

    PetscCall(trace.diagnose(t));

    PetscCheck(std::abs(point_n.z() - L) < z_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle escaped mirror! z = %.6e, allowed = %.6e", point_n.z() - L, z_max);

    PetscReal r = std::hypot(point_n.x() - Rc, point_n.y() - Rc);

    PetscCheck(r < r_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle escaped radial well! r = %.6e, allowed = %.6e", r, r_max);
  }

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
