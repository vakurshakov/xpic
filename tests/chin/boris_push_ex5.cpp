#include "common.h"

// clang-format off
static char help[] =
  "Recreation of published results, see https://doi.org/10.1016/j.jcp.2022.111422  \n"
  "Here we are testing the electron drift in a time-dependent electric and static  \n"
  "magnetic field (described by E0 and B0 parameters) using a different processing \n"
  "algorithms. None that since electric field is on, only \"EB\" algorithms can be \n"
  "used.\n";
// clang-format on

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

  Point point{r0, v0};
  auto particles = prepare_electron(point);

  dt = 0.5;
  geom_nt = 200;

  SyncFile output(get_outputfile(__FILE__, chin_scheme_id));
  output() << "t       x       y       z       \n";
  output() << "[1/wpe] [c/wpe] [c/wpe] [c/wpe] \n";

  BorisPush push;

  if (chin_scheme_id.ends_with("LF"))
    point.r -= (dt / 2.0) * point.p;

  for (PetscInt t = 0; t < geom_nt; ++t) {
    auto interpolated_fields = [t](const Vector3R& /* r */) {
      return std::make_pair(E0 * static_cast<PetscReal>(t) * dt, B0);
    };

    output() << t * dt << " " << point.r << "\n";
    process_impl(chin_scheme_id, push, point, *particles, interpolated_fields);
  }

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
