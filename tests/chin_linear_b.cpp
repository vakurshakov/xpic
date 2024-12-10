#include "chin_common.h"

#define CHIN_SCHEME_ID            BLF
#define CHIN_SCHEME_ID_STR        STR(CHIN_SCHEME_ID)
#define CHIN_SCHEME_OUTPUT        "./tests/chin_linear_b_" CHIN_SCHEME_ID_STR ".txt"
#define CHIN_SCHEME_PROCESS(PUSH) CAT(PUSH.process_, CHIN_SCHEME_ID)

constexpr Vector3R r0(0, 0, 0);
constexpr Vector3R v0(0, 0, 2);

constexpr PetscReal B_uniform = 100;
constexpr PetscReal B_gradient = 25;

Vector3R get_magnetic_field(const Vector3R& r);

int main()
{
  Point point{r0, v0};
  Particles_up particles = prepare_electron(point);

  dt = 0.5;
  geom_nt = 2'500;

  SyncFile output(CHIN_SCHEME_OUTPUT);
  output() << "t       x       y       z       \n";
  output() << "[1/wpe] [c/wpe] [c/wpe] [c/wpe] \n";

  if (std::string(CHIN_SCHEME_ID_STR).ends_with("LF"))
    point.r -= (dt / 2.0) * point.p;

  for (PetscInt t = 0; t < geom_nt; ++t) {
    /// @warning Note, that we use tmp_r for BLF and B1A schemes.
    /// Numerical scheme can be easily screwed with simple `point.r`.
    const Vector3R tmp_r = point.r + point.p * dt;

    BorisPush push(dt, Vector3R(), get_magnetic_field(tmp_r));

    output() << t * dt << " " << point.r << "\n";
    CHIN_SCHEME_PROCESS(push)(point, *particles);
  }
}


Vector3R get_magnetic_field(const Vector3R& r)
{
  return {(B_uniform - B_gradient * r.y()), 0, 0};
}
