#include "common.h"

static char help[] =
  "Here we are testing the electron drift in a time-dependent electric    \n"
  "and static magnetic field (described by E0 and B0 parameters) using a  \n"
  "different processing algorithms. None that since electric field is on, \n"
  "only \"EB\" algorithms can be used. This is a recreation of published  \n"
  "results, see https://doi.org/10.1016/j.jcp.2022.111422 \n";

constexpr Vector3R E0(0.0, -2.0, 0.0);
constexpr Vector3R B0(100.0, 0.0, 0.0);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  std::string chin_scheme_id;
  PetscCall(get_id(chin_scheme_id));

  constexpr Vector3R r0(0.0, 0.0, 0.0);
  constexpr Vector3R v0(0.0, 0.0, 0.1);

  Point point(r0, v0);
  auto particles = prepare_electron(point);

  dt = 0.5;
  geom_nt = 200;
  diagnose_period = geom_nt;

  PointTrace trace(__FILE__, chin_scheme_id, point);

  BorisPush push;
  push.set_qm(-1.0);

  if (chin_scheme_id.ends_with("LF"))
    push.update_r(-dt / 2.0, point);

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    auto interpolated_fields = [t](const Vector3R& /* r */) {
      return std::make_pair(E0 * (t * dt), B0);
    };

    PetscCall(trace.diagnose(t));
    process_impl(chin_scheme_id, push, point, interpolated_fields);
  }

  PetscCall(compare_temporal(__FILE__, chin_scheme_id + ".txt"));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
