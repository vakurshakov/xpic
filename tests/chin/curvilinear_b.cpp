#include "common.h"

// clang-format off
/// @note Only "B" schemes can be used since `Omega * dt >> 1.0`.
#define CHIN_SCHEME_ID     B1B
#define CHIN_SCHEME_ID_STR STR(CHIN_SCHEME_ID)
#define CHIN_SCHEME_OUTPUT "./tests/chin_output/chin_curvilinear_b_" CHIN_SCHEME_ID_STR ".txt"
#define CHIN_SCHEME_PROCESS CAT(process_, CHIN_SCHEME_ID)
// clang-format on

constexpr Vector3R r0(0, 10, 0);
constexpr Vector3R v0(0.16, 1, 0);

constexpr PetscReal B_coeff = 800;
constexpr Vector3R B_center(10, 10, 0);

InterpolationResult get_magnetic_field(const Vector3R& r);

int main()
{
  Point point{r0, v0};
  Particles_up particles = prepare_electron(point);

  dt = 0.16;
  geom_nt = 1000;

  Vector3R check_mean_v;

  SyncFile output(CHIN_SCHEME_OUTPUT);
  output() << "t       x       y       z       \n";
  output() << "[1/wpe] [c/wpe] [c/wpe] [c/wpe] \n";

  BorisPush push;

  if (std::string(CHIN_SCHEME_ID_STR).ends_with("LF"))
    point.r -= (dt / 2.0) * point.p;

  for (PetscInt t = 0; t < geom_nt; ++t) {
    output() << t * dt << " " << point.r << "\n";
    CHIN_SCHEME_PROCESS(push, point, *particles, get_magnetic_field);

    check_mean_v += point.p / static_cast<PetscReal>(geom_nt);
  }

  Vector3R B_p{0, -B_coeff / r0.length(), 0};
  Vector3R B_grad{B_coeff / r0.squared(), 0, 0};

  /// @note v_{\grad(B)} = mv_{\perp}^2 / (2qB) * [B x \grad(B)] / B^2;
  Vector3R v_drift = particles->mass(point) *
    (POW2(v0.x()) + 2.0 * POW2(v0.y())) /
    (2.0 * particles->charge(point) * B_p.length()) * B_p.cross(B_grad) /
    B_p.squared();

  assert(check_mean_v.dot(v_drift) > 0.0);
}

InterpolationResult get_magnetic_field(const Vector3R& r)
{
  Vector3R cr = r - B_center;
  PetscReal rr = cr.length();
  PetscReal ra = std::atan2(cr.y(), cr.x());

  PetscReal B_theta = B_coeff / rr;

  return std::make_pair(Vector3R{},
    Vector3R{
      -std::sin(ra) * B_theta,
      +std::cos(ra) * B_theta,
      0.0,
    });
}
