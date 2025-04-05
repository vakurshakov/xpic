#include "common.h"

static char help[] =
  "Test of electron gyration in a uniform magnetic field using   \n"
  "different process algorithms and comparing several quantities \n"
  "with theory. This is a recreation of published results, see   \n"
  "https://doi.org/10.1016/j.jcp.2022.111422 \n";

constexpr Vector3R B0(0.0, 0.0, 2.0);

InterpolationResult get_magnetic_field(const Vector3R& r);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  std::string chin_scheme_id;
  PetscCall(get_id(chin_scheme_id));

  constexpr Vector3R r0(0.5, 0.0, 0.0);
  constexpr Vector3R v0(0.0, 1.0, 0.0);

  Point point(r0, v0);
  auto particles = prepare_electron(point);

  dt = M_PI / 4.0;
  geom_nt = 50'000;
  diagnose_period = geom_nt / 4;

  PetscReal check_counter_clockwise = 0.0;
  PetscReal check_mean_radius = 0.0;
  Vector3R check_mean_coord;

  // It is not good for `skip` to be even as we wouldn't see the gyration
  PointTrace trace(__FILE__, chin_scheme_id, point, 321);
  BorisPush push;

  if (chin_scheme_id.ends_with("LF"))
    push.update_r(-dt / 2.0, point, *particles);

  // This `omega` is not a cyclotron frequency, but Chin's version of it
  // Since magnetic field is constant, `theta` and `omega` is constant too
  PetscReal omega = B0.length();
  PetscReal theta = omega * dt;
  PetscReal rg = point.p.length() / omega;

  PetscReal Rg = get_effective_larmor(chin_scheme_id, rg, theta);
  Vector3R rc = get_center_offset(chin_scheme_id, rg, theta);

  for (PetscInt t = 0; t < geom_nt; ++t) {
    const Vector3R old_r = point.r;

    PetscCall(trace.diagnose(t));
    process_impl(chin_scheme_id, push, point, *particles, get_magnetic_field);

    // clang-format off
    update_counter_clockwise(old_r, point.r, check_counter_clockwise);
    check_mean_radius += (point.r - rc).length() / static_cast<PetscReal>(geom_nt);
    check_mean_coord += point.r / static_cast<PetscReal>(geom_nt);
    // clang-format on
  }
  PetscCheck(check_counter_clockwise * omega > 0.0, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Electron must rotate counter clockwise. Result ccw count: %f, chin omega: %f", check_counter_clockwise, omega);

  // Checking that magnetic field doesn't do any work on particle
  PetscReal new_rg = point.p.length() / omega;
  PetscCheck(equal_tol(new_rg, rg, 1e-10), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "In uniform field, gyration radius shouldn't change. Result new: %f, old: %f", new_rg, rg);

  PetscCheck(equal_tol(check_mean_radius, Rg, 1e-5), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Mean value of gyration radius should match theory. Result mean: %f, theory: %f", check_mean_radius, Rg);

  PetscCheck(equal_tol(check_mean_coord, rc, 1e-5), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Mean value of gyration center should match theory. Result mean: (%f, %f, %f), theory: (%f, %f, %f)", REP3_A(check_mean_coord), REP3_A(rc));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

InterpolationResult get_magnetic_field(const Vector3R& /* r */)
{
  return std::make_pair(Vector3R{}, B0);
}
