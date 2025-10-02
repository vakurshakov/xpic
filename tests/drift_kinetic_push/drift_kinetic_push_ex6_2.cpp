#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test: magnetic mirror and electric potential along axis.\n"
  "Particle should be reflected at mirror points, center stays\n"
  "between plugs.\n";

using namespace quadratic_magnetic_mirror;

constexpr PetscReal phi = 2.0;

void get_fields(const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = Vector3R{0, 0, -phi * M_PI / L * std::sin(M_PI * r.z() / L)};
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

  constexpr PetscReal v_perp = 1.0;
  constexpr PetscReal v_par = 2.0;
  constexpr Vector3R r0(0.0, 0.0, 0.0);
  constexpr Vector3R v0(v_perp, 0.0, v_par);
  Point point_init(r0, v0);

  PointByField point_n(point_init, {0.0, 0.0, B_min}, m);

  auto id = std::format("omega_dt_{:.1f}", omega_dt);
  PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 123);

  DriftKineticPush push;
  push.set_qm(q / m);
  push.set_mp(m);
  push.set_fields_callback(get_fields);

#if 0
  auto get_v_par2_critical = [v_perp](PetscReal phi) {
    return v_perp * v_perp * (B_max / B_min - 1.0) + 2.0 * q * phi / m;
  };

  PetscReal v_par2_crit_np = std::sqrt(get_v_par2_critical(0));
  PetscReal v_par2_crit_wp = std::sqrt(get_v_par2_critical(phi));

  LOG("Critical v-parallel value (no potential): {}", v_par2_crit_np);
  LOG("Critical v-parallel value (with potential): {}", v_par2_crit_wp);
#endif

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
