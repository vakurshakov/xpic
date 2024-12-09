#include "src/interfaces/particles.h"
#include "src/algorithms/boris_push.h"
#include "src/utils/sync_file.h"
#include "tests/common.h"

constexpr Vector3R E0(0.0, 0.0, 0.0);
constexpr Vector3R B0(0.0, 0.0, 2.0);
constexpr Vector3R v0(0.0, 1.0, 0.0);
constexpr Vector3R r0(0.5, 0.0, 0.0);

#define CHIN_SCHEME_ID            M1B
#define CHIN_SCHEME_OUTPUT        "./tests/chin_gyration_" STR(CHIN_SCHEME_ID) ".txt"
#define CHIN_SCHEME_PROCESS(PUSH) CAT(PUSH.process_, CHIN_SCHEME_ID)

Vector3R get_center_offset(PetscReal rg, PetscReal theta);

int main()
{
  World world;
  SortParameters parameters;
  parameters.q = physical_constants::e * (-1.0);
  parameters.m = physical_constants::me;

  interfaces::Particles particles(world, parameters);

  Point point{r0, v0};
  particles.add_particle(point);

  dt = std::numbers::pi / 4.0;
  geom_nt = 100'000;

  PetscReal check_counter_clockwise = 0.0;
  Vector3R check_mean_r;

  SyncFile output(CHIN_SCHEME_OUTPUT);
  output() << "t       x       y       z       \n";
  output() << "[1/wpe] [c/wpe] [c/wpe] [c/wpe] \n";

  BorisPush push(dt, E0, B0);

  if (std::string(STR(CHIN_SCHEME_ID)).ends_with("LF"))
    point.r -= (dt / 2.0) * point.p;

  /// @note This `omega` is not a cyclotron frequency, but Chin's version of it.
  PetscReal omega = push.get_omega(point, particles);
  assert(omega == 2.0);

  /// @note Since magnetic field is constant, `theta` and `omega` is constant too.
  PetscReal theta = push.get_theta(point, particles);

  for (PetscInt t = 0; t < geom_nt; ++t) {
    const Vector3R old_r = point.r;

    output() << t * dt << " " << point.r << "\n";
    CHIN_SCHEME_PROCESS(push)(point, particles);

    update_counter_clockwise(old_r, point.r, check_counter_clockwise);
    check_mean_r += point.r / static_cast<PetscReal>(geom_nt);
  }
  assert(check_counter_clockwise * omega > 0.0);

  PetscReal rg = point.p.length() / omega;
  assert(equal_tol(rg, r0.length(), 1e-10));

  /// @todo Implement O(theta) check here
  Vector3R rc = get_center_offset(rg, theta);
  assert(equal_tol(check_mean_r, rc, 1e-6));
}


Vector3R get_center_offset(PetscReal rg, PetscReal theta)
{
  auto id = std::string(STR(CHIN_SCHEME_ID));

  /// @note If `id == "MLF"`, then `rc[Y]` left unchanged (and is equal to 0).
  Vector3R rc;

  if (id.starts_with("M")) {
    PetscReal Rg = rg * (theta / 2.0) / std::sin(theta / 2.0);
    rc[X] = rg - Rg * std::cos(theta / 2.0);
    rc[Y] = rg * theta / 2.0;
  }

  if (id.starts_with("B")) {
    PetscReal sin_tb = theta /*               */ / (1.0 + POW2(theta) / 4.0);
    PetscReal cos_tb = (1.0 - POW2(theta) / 4.0) / (1.0 + POW2(theta) / 4.0);
    rc[X] = 0.0;
    rc[Y] = rg * (1 - cos_tb) / sin_tb;  // tan(theta_b / 2) = theta / 2;
  }

  if (id.ends_with("A"))
    rc[Y] *= (-1.0);

  return rc;
}
