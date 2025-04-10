#include "common.h"

static char help[] =
  "Here we are testing the electron drift in crossed electric and \n"
  "magnetic fields using a different process algorithms are used. \n"
  "None that since electric field is on, only \"EB\" algorithms   \n"
  "can be used. This is a recreation of published results, see    \n"
  "https://doi.org/10.1016/j.jcp.2022.111422 \n";

constexpr Vector3R E0(0, 0, 1);
constexpr Vector3R B0(250, 0, 0);

InterpolationResult interpolated_fields(const Vector3R& r);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  std::string chin_scheme_id;
  PetscCall(get_id(chin_scheme_id));

  constexpr Vector3R r0(0.0, 0.0, 0.0);
  constexpr Vector3R v0(0.1, 0.0, 0.4);

  Point point(r0, v0);
  auto particles = prepare_electron(point);

  dt = 0.1975;
  geom_nt = 500;
  diagnose_period = geom_nt;

  PointTrace trace(__FILE__, chin_scheme_id, point, 3);
  BorisPush push;

  if (chin_scheme_id.ends_with("LF"))
    push.update_r(-dt / 2.0, point, *particles);

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    PetscCall(trace.diagnose(t));
    process_impl(chin_scheme_id, push, point, *particles, interpolated_fields);
  }

  PetscCall(compare_temporal(__FILE__, chin_scheme_id + ".txt"));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

InterpolationResult interpolated_fields(const Vector3R& /* r */)
{
  return std::make_pair(E0, B0);
}
