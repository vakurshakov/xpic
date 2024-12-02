#include "src/interfaces/particles.h"
#include "src/algorithms/boris_push.h"
#include "tests/common.h"

constexpr Vector3R E0(0.0, 0.001, 0.0);
constexpr Vector3R B0(0.0, 0.000, 0.2);
constexpr Vector3R r0(0.0, -0.1, 0.0);
constexpr Vector3R v0(0.02, 0.0, 0.0);

int main(int argc, char** argv)
{
  World world;
  Sort_parameters parameters;
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

  const Vector3R v_E = E0.cross(B0) / B0.squared();

  for (PetscReal factor : time_factors) {
    PetscReal dt = factor / std::abs(Omega);
    geom_nt = TO_STEP(geom_t, dt);

    PetscReal check_counter_clockwise = 0.0;
    Vector3R check_drift_coord;

    point.r = r0;
    point.p = v0;

    for (PetscInt t = 0; t < geom_nt; ++t) {
      const Vector3R old_r = point.r;

      Boris_push push(dt, E0, B0);
      push.process_rel(point, particles);

      update_counter_clockwise(old_r, point.r, check_counter_clockwise);
      check_drift_coord += ((point.r - r0) - v_E * (t * dt)) / (PetscReal)geom_nt;
    }
    assert(check_counter_clockwise * Omega < 0.0);

    PetscReal check_parallel = check_drift_coord.parallel_to(v_E).length();
    assert(check_parallel < 1e-1);

    /// @todo Why mean coordinate of transverse motion is not zero?
    PetscReal check_transverse = check_drift_coord.transverse_to(v_E).length();
    assert(check_transverse < 1e-1);
  }
}
