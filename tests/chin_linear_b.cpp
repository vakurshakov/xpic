#include "chin_common.h"

// clang-format off
/// @note Only "B" schemes can be used since `Omega * dt >> 1.0`.
#define CHIN_SCHEME_ID      BLF
#define CHIN_SCHEME_ID_STR  STR(CHIN_SCHEME_ID)
#define CHIN_SCHEME_OUTPUT  "./tests/chin_output/chin_linear_b_" CHIN_SCHEME_ID_STR ".txt"
#define CHIN_SCHEME_PROCESS CAT(process_, CHIN_SCHEME_ID)
// clang-format on

constexpr Vector3R r0(0, 0, 0);
constexpr Vector3R v0(0, 0, 2);

constexpr PetscReal B_uniform = 100;
constexpr PetscReal B_gradient = 25;

InterpolationResult get_magnetic_field(const Vector3R& r);

int main()
{
  Point point{r0, v0};
  Particles_up particles = prepare_electron(point);

  dt = 0.5;
  geom_nt = 2500;

  SyncFile output(CHIN_SCHEME_OUTPUT);
  output() << "t       x       y       z       \n";
  output() << "[1/wpe] [c/wpe] [c/wpe] [c/wpe] \n";

  if (std::string(CHIN_SCHEME_ID_STR).ends_with("LF"))
    point.r -= (dt / 2.0) * point.p;

  BorisPush push;

  for (PetscInt t = 0; t < geom_nt; ++t) {
    output() << t * dt << " " << point.r << "\n";
    CHIN_SCHEME_PROCESS(push, point, *particles, get_magnetic_field);
  }
}

InterpolationResult get_magnetic_field(const Vector3R& r)
{
  return std::make_pair(Vector3R{}, Vector3R{(B_uniform - B_gradient * r.y()), 0, 0});
}
