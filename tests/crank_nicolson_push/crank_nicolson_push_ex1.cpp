#include "crank_nicolson_push.h"

static char help[] =
  "Test of electron gyration in a uniform magnetic field using    \n"
  "Crank-Nicolson integrator for different values of 'Omega * dt' \n"
  "and comparing some quantities, e.g. larmor radius, with theory.\n";

constexpr Vector3R B0(0.0, 0.0, 2.0);

void get_magnetic_field(const Vector3R& xh, Vector3R& E_p, Vector3R& B_p);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  constexpr Vector3R r0(0.5, 0.0, 0.0);
  constexpr Vector3R v0(0.0, 1.0, 0.0);
  Point point_n(r0, v0);

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  std::string id = std::format("omega_dt_{:.1f}", omega_dt);

  dt = omega_dt / B0.length();
  geom_nt = 100'000;
  diagnose_period = geom_nt / 4;

  PetscReal check_counter_clockwise = 0.0;
  PetscReal check_mean_radius = 0.0;
  Vector3R check_mean_coord;

  PointTrace trace(__FILE__, id, point_n, 543);

  CrankNicolsonPush push;
  push.set_qm(-1.0);
  push.set_fields_callback(get_magnetic_field);

  PetscReal omega = -1.0 * B0.length();
  PetscReal rg = v0.length() / std::abs(omega);

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const Point point_0(point_n);

    PetscCall(trace.diagnose(t));
    push.process(dt, point_n, point_0);

    update_counter_clockwise(point_0.r, point_n.r, B0, check_counter_clockwise);
    check_mean_radius += point_n.r.length() / geom_nt;
    check_mean_coord += point_n.r / geom_nt;
  }

  // Though this quantity isn't quite correct for particle gyration with `Omega * dt >> 1`
  PetscCheck(check_counter_clockwise * omega < 0.0, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Electron must rotate counter clockwise. Result ccw count: %f, omega: %f", check_counter_clockwise, omega);

  PetscReal old_E = v0.squared();
  PetscReal new_E = point_n.p.squared();
  PetscCheck(equal_tol(new_E, old_E, PETSC_SMALL), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Particle energy can not be changed in a uniform magnetic field. Result energy new: %.5e, old: %.5e", new_E, old_E);

  PetscCheck(equal_tol(check_mean_radius, rg, 1e-5), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Mean value of gyration radius should match theory. Result mean: %.5f, theory: %.5f", check_mean_radius, rg);

  PetscCheck(equal_tol(check_mean_coord, 0.0, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Mean value of gyration center should match theory. Result mean: (%.4f, %.4f, %.4f), theory: (%.4f, %.4f, %.4f)", REP3_A(check_mean_coord), REP3(0.0));

  PetscCall(compare_temporal(__FILE__, id + ".txt"));

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}

void get_magnetic_field(const Vector3R&, Vector3R&, Vector3R& B_p)
{
  B_p = B0;
}
