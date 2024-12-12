#include "chin_common.h"

// clang-format off
#define CHIN_SCHEME_ID      C2A
#define CHIN_SCHEME_ID_STR  STR(CHIN_SCHEME_ID)
#define CHIN_SCHEME_OUTPUT  "./tests/chin_output/chin_uniform_b_" CHIN_SCHEME_ID_STR ".txt"
#define CHIN_SCHEME_PROCESS CAT(process_, CHIN_SCHEME_ID)
// clang-format on

constexpr Vector3R B0(0.0, 0.0, 2.0);
constexpr Vector3R r0(0.5, 0.0, 0.0);
constexpr Vector3R v0(0.0, 1.0, 0.0);

InterpolationResult get_magnetic_field(const Vector3R& r);

int main()
{
  Point point{r0, v0};
  Particles_up particles = prepare_electron(point);

  dt = std::numbers::pi / 4.0;
  geom_nt = 100'000;

  PetscReal check_counter_clockwise = 0.0;
  PetscReal check_mean_radius = 0.0;
  Vector3R check_mean_coord;

  SyncFile output(CHIN_SCHEME_OUTPUT);
  output() << "t       x       y       z       \n";
  output() << "[1/wpe] [c/wpe] [c/wpe] [c/wpe] \n";

  BorisPush push;

  if (std::string(CHIN_SCHEME_ID_STR).ends_with("LF"))
    point.r -= (dt / 2.0) * point.p;

  /// @note This `omega` is not a cyclotron frequency, but Chin's version of it.
  /// @note Since magnetic field is constant, `theta` and `omega` is constant too.
  PetscReal omega = B0.length();
  PetscReal theta = omega * dt;
  PetscReal rg = point.p.length() / omega;

  PetscReal Rg = get_effective_larmor(CHIN_SCHEME_ID_STR, rg, theta);
  Vector3R rc = get_center_offset(CHIN_SCHEME_ID_STR, rg, theta);

  for (PetscInt t = 0; t < geom_nt; ++t) {
    const Vector3R old_r = point.r;

    output() << t * dt << " " << point.r << "\n";
    CHIN_SCHEME_PROCESS(push, point, *particles, get_magnetic_field);

    update_counter_clockwise(old_r, point.r, check_counter_clockwise);
    check_mean_radius +=
      (point.r - rc).length() / static_cast<PetscReal>(geom_nt);
    check_mean_coord += point.r / static_cast<PetscReal>(geom_nt);
  }
  assert(check_counter_clockwise * omega > 0.0);

  /// @note Checking that magnetic field doesn't do any work on particle.
  PetscReal new_rg = point.p.length() / omega;
  assert(equal_tol(new_rg, rg, 1e-10));

  /// @todo Implement O(theta) check here
  assert(equal_tol(check_mean_radius, Rg, 1e-10));
  assert(equal_tol(check_mean_coord, rc, 1e-5));
}

InterpolationResult get_magnetic_field(const Vector3R& /* r */)
{
  return std::make_pair(Vector3R{}, B0);
}
