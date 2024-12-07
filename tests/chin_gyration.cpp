#include "src/interfaces/particles.h"
#include "src/algorithms/boris_push.h"
#include "src/utils/sync_file.h"
#include "tests/common.h"

#include <bitset>

constexpr Vector3R E0(0.0, 0.0, 0.0);
constexpr Vector3R B0(0.0, 0.0, 2.0);
constexpr Vector3R v0(0.0, 1.0, 0.0);
constexpr Vector3R r0(0.5, 0.0, 0.0);

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
  geom_nt = 3;

  Vector3R ri[3];
  Vector3R vi[3];
  BorisPush push(dt, E0, B0);

  SyncFile output("./tests/chin_gyration.txt");

  for (PetscInt t = 0; t < geom_nt; ++t)
  {
    ri[t % 3] = point.r;
    vi[t % 3] = point.p;

    output() << point.r << "\n";
    output.flush();

    push.process_M1B(point, particles);
  }

  /// @note This `omega` is not a cyclotron frequency, but Chin's version of it.
  PetscReal omega = push.get_omega(point, particles);
  assert(omega == 2.0);

  assert((ri[1] - ri[0]).cross(ri[2] - ri[1]).dot(B0) * omega > 0.0);
  assert((vi[1] - vi[0]).cross(vi[2] - vi[1]).dot(B0) * omega > 0.0);

  /// @note Since magnetic field is constant, `theta` and `omega` is constant too.
  PetscReal theta = push.get_theta(point, particles);

  PetscReal rg = point.p.length() / omega;
  PetscReal Rg = rg * (theta / 2.0) / std::sin(theta / 2.0);

  Vector3R rc{
    rg * (1.0 - (theta / 2.0) / std::tan(theta / 2.0)),
    rg * theta / 2.0,
    0.0,
  };

  assert(equal_tol(rg, r0.length(), 1e-10));
  assert(equal_tol((ri[2] - rc).length(), (ri[0] - rc).length(), 1e-10));
}
