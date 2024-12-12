#include "chin_common.h"

// clang-format off
/// @note Since electric field is on, only "EB" algorithms can be used
#define CHIN_SCHEME_ID      EB2B
#define CHIN_SCHEME_ID_STR  STR(CHIN_SCHEME_ID)
#define CHIN_SCHEME_OUTPUT  "./tests/chin_output/chin_crossed_gradient_" CHIN_SCHEME_ID_STR ".txt"
#define CHIN_SCHEME_PROCESS CAT(process_, CHIN_SCHEME_ID)
// clang-format on

constexpr Vector3R r0(0.0, -1.0, 0.0);
constexpr Vector3R v0(0.1, 0.01, 0.0);

InterpolationResult get_interpolated_fields(const Vector3R& r);

int main()
{
  Point point{r0, v0};
  Particles_up particles = prepare_electron(point);

  // dt = 2.0 * std::numbers::pi / 20.0;
  dt = 2.1 * std::numbers::pi;

  geom_t = 1'000;
  geom_nt = TO_STEP(geom_t, dt);

  SyncFile output(CHIN_SCHEME_OUTPUT);
  output() << "t       x       y       z       \n";
  output() << "[1/wpe] [c/wpe] [c/wpe] [c/wpe] \n";

  BorisPush push;

  if (std::string(CHIN_SCHEME_ID_STR).ends_with("LF"))
    point.r -= (dt / 2.0) * point.p;

  for (PetscInt t = 0; t < geom_nt; ++t) {
    output() << t * dt << " " << point.r << "\n";
    CHIN_SCHEME_PROCESS(push, point, *particles, get_interpolated_fields);
  }
}

InterpolationResult get_interpolated_fields(const Vector3R& r)
{
  static constexpr PetscReal E_coeff = 0.1;
  static constexpr PetscReal B_coeff = 1.0;

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
