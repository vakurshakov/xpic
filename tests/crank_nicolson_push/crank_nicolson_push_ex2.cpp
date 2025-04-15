#include "crank_nicolson_push.h"

static char help[] =
  "Test of electron drift in crossed electric and magnetic fields using \n"
  "Crank-Nicolson integrator for different values of 'Omega * dt' with  \n"
  "checks of energy conservation and drift velocity. Clockwise rotation \n"
  "isn't checked since we will assume the limit of 'Omega * dt >> 1'.   \n";


constexpr Vector3R E0(0, 0, 1);
constexpr Vector3R B0(20, 0, 0);

void interpolated_fields(const Vector3R& r, Vector3R& E_p, Vector3R& B_p);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  constexpr Vector3R r0(0.5, 0.0, 0.0);
  constexpr Vector3R v0(0.0, 1.0, 0.0);
  Point point_n(r0, v0);

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  std::string id = std::format("OmegaDt_{:.1f}", omega_dt);

  PetscReal omega = B0.length();
  dt = omega_dt / omega;
  geom_t = 8000 * (2.0 * M_PI / omega);
  geom_nt = ROUND_STEP(geom_t, dt);
  diagnose_period = geom_nt;

  PetscReal check_energy_conservation = 0.0;
  Vector3R check_drift_velocity;

  PointTrace trace(__FILE__, id, point_n, geom_nt / 123);

  CrankNicolsonPush push;
  push.set_qm(-1.0);
  push.set_fields_callback(interpolated_fields);

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const Point point_0(point_n);

    PetscCall(trace.diagnose(t));
    push.process(dt, point_n, point_0);

    check_energy_conservation +=
      (0.5 * (point_n.p.squared() - point_0.p.squared()) +
        0.5 * (point_n.p + point_0.p).dot(E0) * dt) /
      geom_nt;

    check_drift_velocity +=
      0.5 * (point_n.p + point_0.p).transverse_to(B0) / geom_nt;
  }

  PetscCheck(equal_tol(check_energy_conservation, 0.0, PETSC_SMALL), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Total energy of electron must remain constant, result δK: %.5e", check_energy_conservation);

  Vector3R v_ExB = E0.cross(B0) / B0.squared();
  PetscCheck(equal_tol(check_drift_velocity, v_ExB, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Drift velocity should match the theory. Result v_ExB new: (%.4f, %.4f, %.4f), theory: (%.4f, %.4f, %.4f)", REP3_A(check_drift_velocity), REP3_A(v_ExB));

  PetscCall(compare_temporal(__FILE__, id + ".txt"));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void interpolated_fields(const Vector3R&, Vector3R& E_p, Vector3R& B_p)
{
  E_p = E0;
  B_p = B0;
}
