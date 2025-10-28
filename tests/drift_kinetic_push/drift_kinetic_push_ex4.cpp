#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test: Magnetic mirror field with radial well.\n"
  "Particle should remain trapped in the well; energy conserved.\n";

using namespace quadratic_magnetic_mirror;

constexpr PetscReal phi = 2.0;
constexpr PetscReal E_phi = 0.3;

void get_fields(const Vector3R&, const Vector3R& pos, Vector3R& E_p,
  Vector3R& B_p, Vector3R& gradB_p)
{
  PetscReal x = pos.x();
  PetscReal y = pos.y();
  PetscReal z = pos.z();

  E_p = Vector3R{E_phi * y, -E_phi * x, -phi * M_PI / L * std::sin(M_PI * z / L)};
  get_mirror_fields(pos, B_p, gradB_p);
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  constexpr Vector3R r0(0.5, 0.0, 0.0);
  constexpr PetscReal v_perp = 0.1;
  constexpr PetscReal v_par = 0.1;
  constexpr Vector3R v0(v_perp, 0.0, v_par);
  PetscReal B_start = get_B(r0.length(), r0.z());
  Point point_init(
    r0 + correction::rho(v0, Vector3R(0.0, 0.0, B_start), q / m), v0);
  PointByField point_n(point_init, {0.0, 0.0, B_start}, 1.0, q / m);

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dt = omega_dt / B_start;
  geom_nt = 10'000;
  diagnose_period = geom_nt / 4;

  auto id = std::format("omega_dt_{:.1f}", omega_dt);
  PointByFieldTrace trace(__FILE__, id, point_n, geom_nt / 1000);

  DriftKineticPush push;
  push.set_qm(q / m);
  push.set_mp(m);
  push.set_fields_callback(get_fields);

  PetscReal z_max = L + 1e-2;
  PetscReal r_max = Rc + 1e-2;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_0 = point_n;
    push.process(dt, point_n, point_0);

    PetscCall(trace.diagnose(t));

    PetscCheck(std::abs(point_n.r.z()) < z_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle escaped mirror! z = %.6e, allowed = %.6e", point_n.r.z(), z_max);

    PetscCheck(point_n.r.length() < r_max, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle escaped radial well! r = %.6e, allowed = %.6e", point_n.r.length(), r_max);
  }

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
