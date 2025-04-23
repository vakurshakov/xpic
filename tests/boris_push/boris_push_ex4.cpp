#include "boris_push.h"

static constexpr char help[] =
  "Here we are testing the electron drift in crossed electric and \n"
  "magnetic fields using a different process algorithms are used. \n"
  "None that since electric field is on, only \"EB\" algorithms   \n"
  "can be used. Energy conservation in electric field, as well as \n"
  "particle drift velocity, are checked. This is a recreation of  \n"
  "published results, see https://doi.org/10.1016/j.jcp.2022.111422 \n";

constexpr Vector3R E0(0, 0, 1);
constexpr Vector3R B0(250, 0, 0);

void interpolated_fields(const Vector3R& r, Vector3R& E_p, Vector3R& B_p);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  std::string chin_scheme_id;
  PetscCall(get_id(chin_scheme_id));

  constexpr Vector3R r0(0.0, 0.0, 0.0);
  constexpr Vector3R v0(0.1, 0.0, 0.4);
  Point point(r0, v0);

  dt = 0.1975;
  geom_nt = 5000;
  diagnose_period = geom_nt;

  PetscReal check_energy_conservation = 0.0;
  PetscReal check_counter_clockwise = 0.0;
  Vector3R check_drift_velocity;

  PointTrace trace(__FILE__, chin_scheme_id, point, 32);

  BorisPush push;
  push.set_qm(-1.0);

  if (chin_scheme_id.ends_with("LF"))
    push.update_r(-dt / 2.0, point);

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const Vector3R old_r = point.r;
    const Vector3R old_v = point.p;

    PetscCall(trace.diagnose(t));
    process_impl(chin_scheme_id, push, point, interpolated_fields);

    check_energy_conservation += //
      (0.5 * (point.p.squared() - old_v.squared()) +
        0.5 * (point.p + old_v).dot(E0) * dt) /
      geom_nt;

    update_counter_clockwise(old_r, point.r, B0, check_counter_clockwise);
    check_drift_velocity += 0.5 * (point.p + old_v).transverse_to(B0) / geom_nt;
  }

  PetscCheck(equal_tol(check_energy_conservation, 0.0, PETSC_SMALL), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Total energy of electron must remain constant, result δK: %.5e", check_energy_conservation);

  PetscReal omega = B0.length();
  PetscCheck(check_counter_clockwise * omega > 0.0, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Electron must rotate counter clockwise. Result ccw count: %f, chin omega: %f", check_counter_clockwise, omega);

  Vector3R v_ExB = E0.cross(B0) / B0.squared();
  PetscCheck(equal_tol(check_drift_velocity, v_ExB, 1e-5), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Drift velocity should match the theory. Result v_ExB new: (%.5f, %.5f, %.5f), theory: (%.5f, %.5f, %.5f)", REP3_A(check_drift_velocity), REP3_A(v_ExB));

  PetscCall(compare_temporal(__FILE__, chin_scheme_id + ".txt"));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void interpolated_fields(const Vector3R&, Vector3R& E_p, Vector3R& B_p)
{
  E_p = E0;
  B_p = B0;
}
