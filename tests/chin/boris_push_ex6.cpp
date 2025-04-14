#include "common.h"

static char help[] =
  "Here we are testing the electron drift in a fields, where both electric   \n"
  "and magnetic has the curvature (described by E_coeff and B_coeff).        \n"
  "Different process algorithms are used. None that since electric field is  \n"
  "on, only \"EB\" algorithms can be used. This is a recreation of published \n"
  "results, see https://doi.org/10.1016/j.jcp.2022.111422 \n";

constexpr PetscReal E_coeff = 0.1;
constexpr PetscReal B_coeff = 1.0;

InterpolationResult interpolated_fields(const Vector3R& r);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  std::string chin_scheme_id;
  PetscCall(get_id(chin_scheme_id));

  constexpr Vector3R r0(0.0, -1.0, 0.0);
  constexpr Vector3R v0(0.1, 0.01, 0.0);

  Point point(r0, v0);
  auto particles = prepare_electron(point);

  // dt = 2.0 * M_PI / 20.0;
  dt = 2.1 * M_PI;
  geom_t = 1000;
  geom_nt = ROUND_STEP(geom_t, dt);
  diagnose_period = geom_nt;

  PointTrace trace(__FILE__, chin_scheme_id, point);

  BorisPush push;
  push.set_qm(-1.0);

  if (chin_scheme_id.ends_with("LF"))
    push.update_r(-dt / 2.0, point);

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    PetscCall(trace.diagnose(t));
    process_impl(chin_scheme_id, push, point, interpolated_fields);
  }

  PetscCall(compare_temporal(__FILE__, chin_scheme_id + ".txt"));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

InterpolationResult interpolated_fields(const Vector3R& r)
{
  PetscReal rr = r.length();

  return std::make_pair(
    Vector3R{
      E_coeff * r.x() / POW3(rr),
      E_coeff * r.y() / POW3(rr),
      0.0,
    },
    Vector3R{
      0.0,
      0.0,
      B_coeff * rr,
    });
}
