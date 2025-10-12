#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test: magnetic mirror. Particle should be reflected at mirror\n"
  "points, center stays between plugs.\n";

constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 4.0;
constexpr PetscReal L = 10.0;

void get_fields(
  const Vector3R&, const Vector3R& rn, Vector3R&, Vector3R& B_p, Vector3R& gradB_p)
{
  auto get_B = [](PetscReal z) {
    return B_min + (B_max - B_min) * (z * z) / (L * L);
  };

  auto get_gradB = [](PetscReal z) {
    return 2.0 * (B_max - B_min) * z / (L * L);
  };

  B_p = Vector3R{0, 0, get_B(rn.z())};
  gradB_p = Vector3R{0, 0, get_gradB(rn.z())};
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dt = omega_dt / B_min;
  geom_nt = 1'000;
  diagnose_period = geom_nt / 4;

  // Starting at the minimum of the magnetic field
  constexpr Vector3R r0(0.0, 0.0, 0.0);
  constexpr PetscReal v_perp = 1.0;
  constexpr PetscReal v_par = 1.73;
  constexpr Vector3R v0(v_perp, 0.0, v_par);
  Point point_init(r0, v0);
  PointByField point_n(point_init, {0.0, 0.0, B_min}, m);

  auto id = std::format("omega_dt_{:.1f}", omega_dt);
  PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 10);

  DriftKineticPush push;
  push.set_qm(q / m);
  push.set_mp(m);
  push.set_fields_callback(get_fields);

  // Maximum value of z-coordinates of the particle
  PetscReal z_max = L + 1e-2;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_0 = point_n;
    push.process(dt, point_n, point_0);

    PetscCall(trace.diagnose(t));

    PetscCheck(std::abs(point_n.r.z()) <= z_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle must not escape magnetic mirror! z = %.6e, allowed max = %.6e", point_n.r.z(), z_max);
  }

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
