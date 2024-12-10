#include "src/interfaces/particles.h"
#include "src/algorithms/boris_push.h"
#include "src/utils/sync_file.h"
#include "tests/common.h"

using InterpolationResult = std::pair<REP2(Vector3R)>;
using Interpolator = InterpolationResult (&)(const Vector3R& r);

void process_M1A(BorisPush& push, Point& point,
  interfaces::Particles& particles, const Interpolator& interpolate)
{
  auto [E_p, B_p] = interpolate(point.r);
  push.update_state(dt, E_p, B_p);
  push.update_vM(point, particles);
  push.update_r(point, particles);
}

void process_M1B(BorisPush& push, Point& point,
  interfaces::Particles& particles, const Interpolator& interpolate)
{
  push.update_r(point, particles);

  auto [E_p, B_p] = interpolate(point.r);
  push.update_state(dt, E_p, B_p);
  push.update_vM(point, particles);
}

void process_MLF(BorisPush& push, Point& point,
  interfaces::Particles& particles, const Interpolator& interpolate)
{
  process_M1B(push, point, particles, interpolate);
}

void process_B1A(BorisPush& push, Point& point,
  interfaces::Particles& particles, const Interpolator& interpolate)
{
  auto [E_p, B_p] = interpolate(point.r);
  push.update_state(dt, E_p, B_p);
  push.update_vB(point, particles);
  push.update_r(point, particles);
}

void process_B1B(BorisPush& push, Point& point,
  interfaces::Particles& particles, const Interpolator& interpolate)
{
  push.update_r(point, particles);

  auto [E_p, B_p] = interpolate(point.r);
  push.update_state(dt, E_p, B_p);
  push.update_vB(point, particles);
}

void process_BLF(BorisPush& push, Point& point,
  interfaces::Particles& particles, const Interpolator& interpolate)
{
  process_B1B(push, point, particles, interpolate);
}

void process_C1A(BorisPush& push, Point& point,
  interfaces::Particles& particles, const Interpolator& interpolate)
{
  auto [E_p, B_p] = interpolate(point.r);
  push.update_state(dt, E_p, B_p);
  push.update_vC(point, particles);
  push.update_r(point, particles);
}

void process_C1B(BorisPush& push, Point& point,
  interfaces::Particles& particles, const Interpolator& interpolate)
{
  push.update_r(point, particles);

  auto [E_p, B_p] = interpolate(point.r);
  push.update_state(dt, E_p, B_p);
  push.update_vC(point, particles);
}

void process_CLF(BorisPush& push, Point& point,
  interfaces::Particles& particles, const Interpolator& interpolate)
{
  process_C1B(push, point, particles, interpolate);
}


Particles_up prepare_electron(const Point& point)
{
  World world;
  SortParameters parameters;
  parameters.q = physical_constants::e * (-1.0);
  parameters.m = physical_constants::me;

  auto particles = std::make_unique<interfaces::Particles>(world, parameters);
  particles->add_particle(point);
  return particles;
}


PetscReal get_effective_larmor(std::string_view id, PetscReal rg, PetscReal theta)
{
  if (id.starts_with("M"))
    return rg * (theta / 2.0) / std::sin(theta / 2.0);
  if (id.starts_with("B"))
    return rg * std::sqrt(1.0 + POW2(theta) / 4.0);
  return rg;
}


Vector3R get_center_offset(std::string_view id, PetscReal rg, PetscReal theta)
{
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

  if (id.starts_with("C")) {
    PetscReal cos_tc = (1.0 - POW2(theta) / 2.0);
    rc[X] = rg * (1.0 - std::sqrt(1.0 - POW2(theta) / 4.0));
    rc[Y] = rg * std::sqrt((1.0 - cos_tc) / 2.0);
  }

  if (id.ends_with("A"))
    rc[Y] *= (-1.0);
  else if (id.ends_with("LF"))
    rc[Y] = 0.0;

  return rc;
}
