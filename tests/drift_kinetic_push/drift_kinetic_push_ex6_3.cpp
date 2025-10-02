#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test: magnetic mirror and constant azimuthal E_phi.\n"
  "Check that particle is pushed toward axis due to ExB drift.\n";

using namespace quadratic_magnetic_mirror;

constexpr PetscReal E_phi = 0.3;

void get_fields(const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  PetscReal x = r.x();
  PetscReal y = r.y();
  PetscReal d = std::hypot(x, y);

  E_p = Vector3R{+E_phi * y / d, -E_phi * x / d, 0};
  B_p = Vector3R{0, 0, get_Bz(r.z())};
  gradB_p = Vector3R{0, 0, get_dBz_dz(r.z())};
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dt = omega_dt / B_min;
  geom_nt = 100'000;
  diagnose_period = geom_nt / 4;

  constexpr PetscReal R0 = 0.8;
  constexpr PetscReal v_perp = 1.0;
  constexpr PetscReal v_par = 1.0;
  constexpr Vector3R r0(R0, 0.0, 0.0);
  constexpr Vector3R v0(v_perp, 0.0, v_par);
  Point point_init(r0, v0);
  PointByField point_n(point_init, {0.0, 0.0, B_min}, m);

  auto id = std::format("omega_dt_{:.1f}", omega_dt);
  PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 123);

  DriftKineticPush push;
  push.set_qm(q / m);
  push.set_mp(m);
  push.set_fields_callback(get_fields);

  PetscReal z_max = L + 1e-2;
  PetscReal r_sum = 0.0;
  PetscReal r_min = +std::numeric_limits<PetscReal>::max();
  PetscReal r_max = -std::numeric_limits<PetscReal>::max();

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_0 = point_n;

    PetscCall(trace.diagnose(t));
    push.process(dt, point_n, point_0);

    PetscReal r_now = std::hypot(point_n.r.x(), point_n.r.y());

    r_sum += r_now / geom_nt;
    r_min = (r_now < r_min) ? r_now : r_min;
    r_max = (r_now > r_max) ? r_now : r_max;

    PetscCheck(std::abs(point_n.r.z()) < z_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle must not escape magnetic mirror! z = %.6e, allowed max = %.6e", point_n.r.z(), z_max);

    PetscCheck(r_now < R0, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle went outside allowed radius! r = %.3e, allowed max = %.3e", r_now, R0);
  }

#if 0
  LOG("Mean radius: {}, min r: {}, max r: {}", r_sum, r_min, r_max);
#endif

  PetscCheck(r_sum < R0, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Particle was not pushed to axis by E_phi: mean r = %.3e, start %.3e", r_sum, R0);

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
