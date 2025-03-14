#include "common.h"

// clang-format off
static char help[] =
  "Recreation of published results, see https://doi.org/10.1016/j.jcp.2022.111422 \n"
  "Here we are testing the electron drift in crossed electric and magnetic fields \n"
  "using a different process algorithms are used. None that since electric field  \n"
  "is on, only \"EB\" algorithms can be used.\n";
// clang-format on

constexpr Vector3R E0(0, 0, 1);
constexpr Vector3R B0(250, 0, 0);

InterpolationResult get_interpolated_fields(const Vector3R& r);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  std::string chin_scheme_id;
  PetscCall(get_id(chin_scheme_id));

  LOG("{}", chin_scheme_id);

  constexpr Vector3R r0(0.0, 0.0, 0.0);
  constexpr Vector3R v0(0.1, 0.0, 0.4);

  Point point{r0, v0};
  Particles_up particles = prepare_electron(point);

  dt = 0.1975;
  geom_nt = 500;

  SyncFile output(get_outputfile(__FILE__, chin_scheme_id));
  output() << "t       x       y       z       \n";
  output() << "[1/wpe] [c/wpe] [c/wpe] [c/wpe] \n";

  BorisPush push;

  if (chin_scheme_id.ends_with("LF"))
    point.r -= (dt / 2.0) * point.p;

  for (PetscInt t = 0; t < geom_nt; ++t) {
    output() << t * dt << " " << point.r << "\n";
    process_impl(chin_scheme_id, push, point, *particles, get_interpolated_fields);
  }

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

InterpolationResult get_interpolated_fields(const Vector3R& /* r */)
{
  return std::make_pair(E0, B0);
}
