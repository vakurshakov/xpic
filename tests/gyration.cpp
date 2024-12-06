#include "src/interfaces/particles.h"
#include "src/algorithms/boris_push.h"
#include "tests/common.h"

constexpr Vector3R E0(0.0, 0.0, 0.0);
constexpr Vector3R B0(0.0, 0.0, 0.2);
constexpr Vector3R r0(0.0, -0.1, 0.0);
constexpr Vector3R v0(0.02, 0.0, 0.0);

int main()
{
  World world;
  SortParameters parameters;
  parameters.q = (-1.0) * physical_constants::e;
  parameters.m = physical_constants::me;

  interfaces::Particles particles(world, parameters);

  Point point;
  particles.add_particle(point);

  PetscReal Omega = parameters.q * B0.length() / parameters.m;
  PetscReal rho = v0.length() / std::abs(Omega);

  geom_t = 100'000;

  std::vector<PetscReal> time_factors{
    0.05,
    0.2,
    1.0,
    2.0,
    5.0,
  };

  for (PetscReal factor : time_factors) {
    PetscReal dt = factor / std::abs(Omega);
    geom_nt = TO_STEP(geom_t, dt);

    /// @todo Gyration mean coordinate diverges with growing `dt`
    PetscReal check_counter_clockwise = 0.0;
    PetscReal check_mean_radius = 0.0;

    point.r = r0;
    point.p = v0;

    for (PetscInt t = 0; t < geom_nt; ++t) {
      const Vector3R old_r = point.r;

      BorisPush push(dt, E0, B0);
      push.process(point, particles);

      update_counter_clockwise(old_r, point.r, check_counter_clockwise);
      check_mean_radius += point.r.length() / static_cast<PetscReal>(geom_nt);
    }
    assert(check_counter_clockwise * Omega < 0.0);

    PetscReal effective_rho = rho * std::sqrt(1 + POW2(Omega * dt) / 4.0);
    assert(std::abs(check_mean_radius / effective_rho - 1.0) < 8e-2 * POW2(Omega * dt));
  }
}
