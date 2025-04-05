#include "common.h"

static char help[] =
  "Here we are testing the electron gyration in a linearly-chaning    \n"
  "magnetic field, described by B_uniform and B_gradient parameters,  \n"
  "using a different process algorithms. None that only \"B\" schemes \n"
  "can be used since `Omega * dt >> 1.0`. This is a recreation of     \n"
  "published results, see https://doi.org/10.1016/j.jcp.2022.111422   \n";

constexpr PetscReal B_uniform = 100;
constexpr PetscReal B_gradient = 25;

InterpolationResult get_magnetic_field(const Vector3R& r);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  std::string chin_scheme_id;
  PetscCall(get_id(chin_scheme_id));

  constexpr Vector3R r0(0, 0, 0);
  constexpr Vector3R v0(0, 0, 2);

  Point point(r0, v0);
  auto particles = prepare_electron(point);

  dt = 0.5;
  geom_nt = 1000;
  diagnose_period = geom_nt;

  PointTrace trace(__FILE__, chin_scheme_id, point, 5);
  BorisPush push;

  if (chin_scheme_id.ends_with("LF"))
    push.update_r(-dt / 2.0, point, *particles);

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    PetscCall(trace.diagnose(t));
    process_impl(chin_scheme_id, push, point, *particles, get_magnetic_field);
  }

  PetscCall(compare_temporal(__FILE__, chin_scheme_id + ".txt"));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

InterpolationResult get_magnetic_field(const Vector3R& r)
{
  return std::make_pair(
    Vector3R{}, Vector3R{(B_uniform - B_gradient * r.y()), 0, 0});
}
