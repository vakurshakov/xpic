#include "chin_common.h"

// clang-format off
#define CHIN_SCHEME_ID      BLF
#define CHIN_SCHEME_ID_STR  STR(CHIN_SCHEME_ID)
#define CHIN_SCHEME_OUTPUT  "./tests/chin_output/chin_linear_b_" CHIN_SCHEME_ID_STR ".txt"
#define CHIN_SCHEME_PROCESS CAT(process_, CHIN_SCHEME_ID)
// clang-format on

constexpr Vector3R r0(0, 0, 0);
constexpr Vector3R v0(0, 0, 2);

constexpr PetscReal B_uniform = 100;
constexpr PetscReal B_gradient = 25;

Vector3R get_magnetic_field(const Vector3R& r);

/// @note Only "B" schemes can be used since `Omega * dt >> 1.0`.
void process_B1A(BorisPush& push, Point& point, interfaces::Particles& particles);
void process_B1B(BorisPush& push, Point& point, interfaces::Particles& particles);
void process_BLF(BorisPush& push, Point& point, interfaces::Particles& particles);

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
    CHIN_SCHEME_PROCESS(push, point, *particles);
  }
}

Vector3R get_magnetic_field(const Vector3R& r)
{
  return {(B_uniform - B_gradient * r.y()), 0, 0};
}

void process_B1A(BorisPush& push, Point& point, interfaces::Particles& particles)
{
  push.update_state(dt, Vector3R{}, get_magnetic_field(point.r));
  push.update_vB(point, particles);
  push.update_r(point, particles);
}

void process_B1B(BorisPush& push, Point& point, interfaces::Particles& particles)
{
  push.update_r(point, particles);
  push.update_state(dt, Vector3R{}, get_magnetic_field(point.r));
  push.update_vB(point, particles);
}

void process_BLF(BorisPush& push, Point& point, interfaces::Particles& particles)
{
  process_B1B(push, point, particles);
}
