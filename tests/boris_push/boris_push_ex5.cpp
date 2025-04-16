#include "boris_push.h"

static char help[] =
  "Here we are testing the electron drift in a time-dependent electric and   \n"
  "static magnetic field (described by E0 and B0 parameters) using different \n"
  "processing algorithms. None that since electric field is on, only \"EB\"  \n"
  "algorithms can be used. Energy conservation in electric field, along with \n"
  "ExB + pol. drift velocity, are checked. This is a recreation of published \n"
  "results, see https://doi.org/10.1016/j.jcp.2022.111422 \n";

constexpr Vector3R E0(0, -2, 0);
constexpr Vector3R B0(100, 0, 0);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  std::string chin_scheme_id;
  PetscCall(get_id(chin_scheme_id));

  constexpr Vector3R r0(0.0, 0.0, 0.0);
  constexpr Vector3R v0(0.0, 0.0, 0.1);
  Point point(r0, v0);

  dt = 0.5;
  geom_nt = 10'000;
  diagnose_period = geom_nt;

  PetscReal check_energy_conservation = 0.0;
  PetscReal check_counter_clockwise = 0.0;
  Vector3R check_drift_velocity;

  PointTrace trace(__FILE__, chin_scheme_id, point, 54);

  BorisPush push;
  push.set_qm(-1.0);

  if (chin_scheme_id.ends_with("LF"))
    push.update_r(-dt / 2.0, point);

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const Vector3R old_r = point.r;
    const Vector3R old_v = point.p;

    auto interpolated_fields = [t](const Vector3R&, Vector3R& E_p, Vector3R& B_p) {
      E_p = E0 * (t * dt);
      B_p = B0;
    };

    PetscCall(trace.diagnose(t));
    process_impl(chin_scheme_id, push, point, interpolated_fields);

    check_energy_conservation += //
      (0.5 * (point.p.squared() - old_v.squared()) +
        0.5 * (point.p + old_v).dot(E0) * t * POW2(dt)) /
      geom_nt;

    update_counter_clockwise(old_r, point.r, B0, check_counter_clockwise);
    check_drift_velocity += 0.5 * (point.p + old_v).transverse_to(B0) / geom_nt;
  }

  PetscCheck(equal_tol(check_energy_conservation, 0.0, 5.0 * PETSC_SMALL), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Total energy of electron must remain constant, result δK: %.5e", check_energy_conservation);

  PetscReal omega = B0.length();
  PetscCheck(check_counter_clockwise * omega > 0.0, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Electron must rotate counter clockwise. Result ccw count: %f, chin omega: %f", check_counter_clockwise, omega);

  // We have to integrate electric field over the `geom_nt` time period
  // (arithmetic series) because it changes linearly in _time_
  PetscReal E_time_int = (PetscReal)(geom_nt + 1) * (0.5 * dt);
  Vector3R v_ExB = E0.cross(B0) / B0.squared() * E_time_int;
  Vector3R h = B0.normalized();
  Vector3R v_pol = (-1.0) * h.cross(E0.cross(h)) / B0.squared();
  Vector3R v_drift = v_ExB + v_pol;
  PetscCheck(equal_tol(check_drift_velocity, v_drift, 1e-6), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Drift velocity should match the theory. Result v_drift new: (%.6f, %.6f, %.6f), theory: (%.6f, %.6f, %.6f)", REP3_A(check_drift_velocity), REP3_A(v_drift));

  PetscCall(compare_temporal(__FILE__, chin_scheme_id + ".txt"));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
