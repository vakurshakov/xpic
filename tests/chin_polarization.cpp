#include "chin_common.h"

// clang-format off
/// @note Since electric field is on, only "EB" algorithms can be used
#define CHIN_SCHEME_ID      EB2B
#define CHIN_SCHEME_ID_STR  STR(CHIN_SCHEME_ID)
#define CHIN_SCHEME_OUTPUT  "./tests/chin_output/chin_polarization_" CHIN_SCHEME_ID_STR ".txt"
#define CHIN_SCHEME_PROCESS CAT(process_, CHIN_SCHEME_ID)
// clang-format on

constexpr Vector3R E0(0.0, -2.0, 0.0);
constexpr Vector3R B0(100.0, 0.0, 0.0);
constexpr Vector3R r0(0.0, 0.0, 0.0);
constexpr Vector3R v0(0.0, 0.0, 0.1);

int main()
{
  Point point{r0, v0};
  Particles_up particles = prepare_electron(point);

  dt = 0.5;
  geom_nt = 200;

  SyncFile output(CHIN_SCHEME_OUTPUT);
  output() << "t       x       y       z       \n";
  output() << "[1/wpe] [c/wpe] [c/wpe] [c/wpe] \n";

  BorisPush push;

  if (std::string(CHIN_SCHEME_ID_STR).ends_with("LF"))
    point.r -= (dt / 2.0) * point.p;

  for (PetscInt t = 0; t < geom_nt; ++t) {
    auto get_interpolated_fields = [t](const Vector3R& /* r */) {
      return std::make_pair(E0 * static_cast<PetscReal>(t) * dt, B0);
    };

    output() << t * dt << " " << point.r << "\n";
    CHIN_SCHEME_PROCESS(push, point, *particles, get_interpolated_fields);
  }
}
