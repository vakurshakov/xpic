#include <cassert>

#include "src/interfaces/particles.h"
#include "src/algorithms/boris_push.h"
#include "src/utils/vector3.h"

constexpr Vector3R E0(0.0, 0.0, 0.0);
constexpr Vector3R B0(0.0, 0.0, 0.2);
constexpr Vector3R r0(0.0, -0.1, 0.0);
constexpr Vector3R v0(0.02, 0.0, 0.0);

void update_clockwise(
  const Vector3R& old_r, const Vector3R& new_r, PetscReal& clockwise);


/// @todo replace with google-tests environment
int main(int argc, char** argv)
{
  World world;
  Sort_parameters parameters;
  parameters.q = (-1.0) * physical_constants::e;
  parameters.m = physical_constants::me;

  geom_nt = 100'000;

  interfaces::Particles particles(world, parameters);

  Point point;
  PetscCall(particles.add_particle(point));

  PetscReal Omega = parameters.q * B0.length() / parameters.m;
  PetscReal rho = v0.length() / std::abs(Omega);

  std::vector<PetscReal> time_factors{
    0.05,
    0.2,
    1.0,
    2.0,
    5.0,
  };

  for (PetscReal factor : time_factors) {
    PetscReal dt = factor / std::abs(Omega);

    PetscReal check_clockwise = 0.0;
    PetscReal check_mean_radius = 0.0;

    point.r = r0;
    point.p = v0;

    for (PetscInt t = 0; t < geom_nt; ++t) {
      const Vector3R old_r = point.r;

      Boris_push push(dt, E0, B0);
      push.process(point, particles);

      update_clockwise(old_r, point.r, check_clockwise);

      check_mean_radius += point.r.length() / geom_nt;
    }
    assert(check_clockwise * Omega < 0.0);

    PetscReal effective_radius = rho * std::sqrt(1 + POW2(Omega * dt) / 4.0);
    assert(std::abs(check_mean_radius - effective_radius) < 1e-1);
  }
}


void update_clockwise(
  const Vector3R& old_r, const Vector3R& new_r, PetscReal& clockwise)
{
  clockwise += (old_r.y() + new_r.y()) * (old_r.x() - new_r.x());
}
