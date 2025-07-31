#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test: Magnetic mirror field with radial well.\n"
  "Particle should remain trapped in the well; energy conserved.\n";

constexpr PetscReal B_min = 1.0;
constexpr PetscReal B_max = 4.0;
constexpr PetscReal L = 10.0;   // half-length of the mirror
constexpr PetscReal Rc = 20.0;  // width of the radial well

PetscReal get_Bz(PetscReal z)
{
  return B_min + (B_max - B_min) * (z * z) / (L * L);
}

PetscReal get_B(PetscReal r, PetscReal z)
{
  return get_Bz(z) * (1.0 + 0.5 * (r * r) / (Rc * Rc));
}

void get_fields(
  const Vector3R& pos, Vector3R&, Vector3R& B_p, Vector3R& gradB_p)
{
  PetscReal x = pos.x();
  PetscReal y = pos.y();
  PetscReal z = pos.z();
  PetscReal r = std::hypot(x, y);

  PetscReal Bz = get_Bz(z);
  PetscReal B = get_B(r, z);

  B_p = Vector3R{0.0, 0.0, B};

  PetscReal dBz_dz = 2.0 * (B_max - B_min) * z / (L * L);
  PetscReal dB_dz = dBz_dz * (1.0 + 0.5 * (r * r) / (Rc * Rc));
  PetscReal dB_dr = Bz * r / (Rc * Rc);

  gradB_p = (r > 1e-10) //
    ? Vector3R{x / r * dB_dr, y / r * dB_dr, dB_dz}
    : Vector3R{0.0, 0.0, dB_dz};
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  constexpr PetscReal v_perp = 1.0;
  constexpr PetscReal v_par = 0.6;
  constexpr Vector3R r0(0.5, 0.0, 0.0);
  constexpr Vector3R v0(v_perp, 0.0, v_par);
  Point point_init(r0, v0);
  PointByField point_n(point_init, {0, 0, get_B(r0.length(), r0.z())}, 1);

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dt = omega_dt / get_B(r0.length(), r0.z());
  geom_nt = 100'000;
  diagnose_period = geom_nt / 4;

  auto id = std::format("omega_dt_{:.1f}", omega_dt);
  PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 123);

  DriftKineticPush push;
  push.set_qm(1.0);
  push.set_mp(1.0);
  push.set_fields_callback(get_fields);

  PetscReal z_max = L + 1e-2;
  PetscReal r_max = Rc + 1e-2;

  const PetscReal old_E = //
    point_n.p_parallel * point_n.p_parallel + //
    point_n.p_perp * point_n.p_perp;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_0 = point_n;
    push.process(dt, point_n, point_0);

    PetscCall(trace.diagnose(t));

    PetscCheck(std::abs(point_n.r.z()) < z_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle escaped mirror! z = %.6e, allowed = %.6e", point_n.r.z(), z_max);

    PetscCheck(point_n.r.length() < r_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle escaped radial well! r = %.6e, allowed = %.6e", point_n.r.length(), r_max);
  }

  const PetscReal new_E = //
    point_n.p_parallel * point_n.p_parallel + //
    point_n.p_perp * point_n.p_perp;

  PetscCheck(equal_tol(new_E, old_E, 1e-6), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Energy not conserved: new = %.10e, old = %.10e", new_E, old_E);

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
