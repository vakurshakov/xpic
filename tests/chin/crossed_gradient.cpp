#include "common.h"

// clang-format off
static char help[] =
  "Recreation of published results, see https://doi.org/10.1016/j.jcp.2022.111422       \n"
  "Here we are testing the electron drift in a fields, where both electric and magnetic \n"
  "has the curvature (described by E_coeff and B_coeff). Different process algorithms   \n"
  "are used. None that since electric field is on, only \"EB\" algorithms can be used.  \n";
// clang-format on

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

  Point point{r0, v0};
  Particles_up particles = prepare_electron(point);

  // dt = 2.0 * std::numbers::pi / 20.0;
  dt = 2.1 * std::numbers::pi;

  geom_t = 1000;
  geom_nt = ROUND_STEP(geom_t, dt);

  SyncFile output(get_outputfile(__FILE__, chin_scheme_id));
  output() << "t       x       y       z       \n";
  output() << "[1/wpe] [c/wpe] [c/wpe] [c/wpe] \n";

  BorisPush push;

  if (chin_scheme_id.ends_with("LF"))
    point.r -= (dt / 2.0) * point.p;

  for (PetscInt t = 0; t < geom_nt; ++t) {
    output() << t * dt << " " << point.r << "\n";
    process_impl(chin_scheme_id, push, point, *particles, interpolated_fields);
  }

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
