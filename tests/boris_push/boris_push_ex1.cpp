#include "boris_push.h"

static constexpr char help[] =
  "Test of electron gyration in a uniform magnetic field using   \n"
  "different process algorithms and comparing several quantities \n"
  "with theory. This is a recreation of published results, see   \n"
  "https://doi.org/10.1016/j.jcp.2022.111422 \n";

constexpr Vector3R B0(0.0, 0.0, 2.0);

void get_magnetic_field(const Vector3R& r, Vector3R& E_p, Vector3R& B_p);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  std::string chin_scheme_id;
  PetscCall(get_id(chin_scheme_id));

  constexpr Vector3R r0(0.5, 0.0, 0.0);
  constexpr Vector3R v0(0.0, 1.0, 0.0);
  Point point(r0, v0);

  dt = M_PI / 4.0;
  geom_nt = 100'000;
  diagnose_period = geom_nt;

  PetscReal check_counter_clockwise = 0.0;
  PetscReal check_mean_radius = 0.0;
  Vector3R check_mean_coord;

  // It is not good for `skip` to be even as we wouldn't see the gyration
  PointTrace trace(__FILE__, chin_scheme_id, point, 543);

  BorisPush push;
  push.set_qm(-1.0);

  if (chin_scheme_id.ends_with("LF"))
    push.update_r(-dt / 2.0, point);

  // This `omega` is not a cyclotron frequency, but Chin's version of it
  // Since magnetic field is constant, `theta` and `omega` is constant too
  PetscReal omega = B0.length();
  PetscReal theta = omega * dt;
  PetscReal rg = point.p.length() / omega;

  PetscReal Rg = get_effective_larmor(chin_scheme_id, rg, theta);
  Vector3R rc = get_center_offset(chin_scheme_id, rg, theta);

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const Vector3R old_r = point.r;

    PetscCall(trace.diagnose(t));
    process_impl(chin_scheme_id, push, point, get_magnetic_field);

    update_counter_clockwise(old_r, point.r, B0, check_counter_clockwise);
    check_mean_radius += (point.r - rc).length() / geom_nt;
    check_mean_coord += point.r / geom_nt;
  }

  PetscCheck(check_counter_clockwise * omega > 0.0, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Electron must rotate counter clockwise. Result ccw count: %f, chin omega: %f", check_counter_clockwise, omega);

  PetscReal old_E = v0.squared();
  PetscReal new_E = point.p.squared();
  PetscCheck(equal_tol(new_E, old_E, PETSC_SMALL), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Particle energy can not be changed in a uniform magnetic field. Result energy new: %.5e, old: %.5e", new_E, old_E);

  PetscCheck(equal_tol(check_mean_radius, Rg, 1e-5), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Mean value of gyration radius should match theory. Result mean: %.5f, theory: %.5f", check_mean_radius, Rg);

  PetscCheck(equal_tol(check_mean_coord, rc, 1e-4), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Mean value of gyration center should match theory. Result mean: (%.4f, %.4f, %.4f), theory: (%.4f, %.4f, %.4f)", REP3_A(check_mean_coord), REP3_A(rc));

  PetscCall(compare_temporal(__FILE__, chin_scheme_id + ".txt"));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void get_magnetic_field(const Vector3R&, Vector3R&, Vector3R& B_p)
{
  B_p = B0;
}
