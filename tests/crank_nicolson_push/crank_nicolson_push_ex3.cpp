#include "crank_nicolson_push.h"

static constexpr char help[] =
  "Here we are testing the electron drift in a time-dependent electric and   \n"
  "static magnetic field (described by E0 and B0 parameters) using Crank-    \n"
  "Nicolson integrator for different values of 'Omega * dt'. Conservation of \n"
  "energy and drift velocity (ExB + pol.) is checked. \n";

constexpr Vector3R E0(0, -2, 0);
constexpr Vector3R B0(100, 0, 0);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  constexpr Vector3R r0(0.0, 0.0, 0.0);
  constexpr Vector3R v0(0.0, 0.0, 0.1);
  Point point_n(r0, v0);

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  std::string id = std::format("omega_dt_{:.1f}", omega_dt);

  PetscReal omega = B0.length();
  dt = omega_dt / omega;
  geom_t = 30'000 * (2.0 * M_PI / omega);
  geom_nt = ROUND_STEP(geom_t, dt);
  diagnose_period = geom_nt;

  PetscReal check_energy_conservation = 0.0;
  Vector3R check_drift_velocity;

  PointTrace trace(__FILE__, id, point_n, geom_nt / 123);

  CrankNicolsonPush push;
  push.set_qm(-1.0);

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const Point point_0(point_n);

    push.set_fields_callback(
      [t](const Vector3R&, const Vector3R&, Vector3R& E_p, Vector3R& B_p) {
        E_p = E0 * (t * dt);
        B_p = B0;
      });

    PetscCall(trace.diagnose(t));
    push.process(dt, point_n, point_0);

    check_energy_conservation +=
      (0.5 * (point_n.p.squared() - point_0.p.squared()) +
        0.5 * (point_n.p + point_0.p).dot(E0) * t * POW2(dt)) /
      geom_nt;

    check_drift_velocity +=
      0.5 * (point_n.p + point_0.p).transverse_to(B0) / geom_nt;
  }

  PetscCheck(equal_tol(check_energy_conservation, 0.0, 5.0 * PETSC_SMALL), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Total energy of electron must remain constant, result δK: %.5e", check_energy_conservation);

  // We have to integrate electric field over the `geom_nt` time period
  // (arithmetic series) because it changes linearly in _time_
  PetscReal E_time_int = (PetscReal)(geom_nt + 1) * (0.5 * dt);
  Vector3R v_ExB = E0.cross(B0) / B0.squared() * E_time_int;
  Vector3R h = B0.normalized();
  Vector3R v_pol = (-1.0) * h.cross(E0.cross(h)) / B0.squared();
  Vector3R v_drift = v_ExB + v_pol;
  PetscCheck(equal_tol(check_drift_velocity, v_drift, 1e-6), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Drift velocity should match the theory. Result v_drift new: (%.6f, %.6f, %.6f), theory: (%.6f, %.6f, %.6f)", REP3_A(check_drift_velocity), REP3_A(v_drift));

  PetscCall(compare_temporal(__FILE__, id + ".txt"));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
