#include "common.h"

// clang-format off
static char help[] =
  "Recreation of published results, see https://doi.org/10.1016/j.jcp.2022.111422 \n"
  "Here we are testing the electron gyration in a linearly-chaning magnetic field,\n"
  "described by B_uniform and B_gradient parameters, using a different process    \n"
  "algorithms. None that only \"B\" schemes can be used since `Omega * dt >> 1.0`.\n";
// clang-format on

constexpr PetscReal B_uniform = 100;
constexpr PetscReal B_gradient = 25;

InterpolationResult get_magnetic_field(const Vector3R& r);

/// @todo Check at least a diff comparison between new `output` and the old one.
int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  std::string chin_scheme_id;
  PetscCall(get_id(chin_scheme_id));

  constexpr Vector3R r0(0, 0, 0);
  constexpr Vector3R v0(0, 0, 2);

  Point point{r0, v0};
  Particles_up particles = prepare_electron(point);

  dt = 0.5;
  geom_nt = 2500;

  SyncFile output(get_outputfile(__FILE__, chin_scheme_id));
  output() << "t       x       y       z       \n";
  output() << "[1/wpe] [c/wpe] [c/wpe] [c/wpe] \n";

  if (chin_scheme_id.ends_with("LF"))
    point.r -= (dt / 2.0) * point.p;

  BorisPush push;

  for (PetscInt t = 0; t < geom_nt; ++t) {
    output() << t * dt << " " << point.r << "\n";
    process_impl(chin_scheme_id, push, point, *particles, get_magnetic_field);
  }

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

InterpolationResult get_magnetic_field(const Vector3R& r)
{
  return std::make_pair(
    Vector3R{}, Vector3R{(B_uniform - B_gradient * r.y()), 0, 0});
}
