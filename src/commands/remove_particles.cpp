#include "remove_particles.h"

#include "src/utils/configuration.h"
#include "src/utils/region_operations.h"

class RemoveParticles::RemoveFromBox {
public:
  RemoveFromBox(const BoxGeometry& geom);
  bool operator()(const Point& point);
  BoxGeometry geom_;
};


class RemoveParticles::RemoveFromCircle {
public:
  RemoveFromCircle(const CircleGeometry& geom);
  bool operator()(const Point& point);
  CircleGeometry geom_;
};


RemoveParticles::RemoveParticles(
  interfaces::Particles& particles, const BoxGeometry& geom)
  : particles_(particles), should_remove_(RemoveFromBox(geom))
{
}

RemoveParticles::RemoveParticles(
  interfaces::Particles& particles, const CircleGeometry& geom)
  : particles_(particles), should_remove_(RemoveFromCircle(geom))
{
}

PetscErrorCode RemoveParticles::execute(timestep_t /* t */)
{
  PetscFunctionBeginUser;
  PetscInt removed_particles = 0;
  removed_energy_ = 0.0;

  const PetscInt Np = particles_.parameters.Np;
  const PetscReal m = particles_.parameters.m;

  /// @note Now we can check only `cell` indices, not the `point` coordinates!
  for (auto& cell : particles_.storage) {
    cell.remove_if([&](const Point& point) {
      if (should_remove_(point)) {
        removed_particles++;
        removed_energy_ += 0.5 * (m * point.p.squared()) * (dx * dy * dz) / Np;
        return true;
      }
      return false;
    });
  }

  if (removed_particles > 0) {
    LOG("  Particles are removed from \"{}\", particles: {}, energy: {}",
      particles_.parameters.sort_name, removed_particles, removed_energy_);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

std::string RemoveParticles::get_particles_name() const
{
  return particles_.parameters.sort_name;
}

PetscReal RemoveParticles::get_removed_energy() const
{
  return removed_energy_;
}


RemoveParticles::RemoveFromBox::RemoveFromBox(const BoxGeometry& geom)
  : geom_(geom)
{
}

bool RemoveParticles::RemoveFromBox::operator()(const Point& point)
{
  /// @todo REUSE region_operations.h AS TEMPLATES
  return  //
    (geom_.min[X] > point.x() || point.x() > geom_.max[X]) ||
    (geom_.min[Y] > point.y() || point.y() > geom_.max[Y]) ||
    (geom_.min[Z] > point.z() || point.z() > geom_.max[Z]);
}


RemoveParticles::RemoveFromCircle::RemoveFromCircle(const CircleGeometry& geom)
  : geom_(geom)
{
}

bool RemoveParticles::RemoveFromCircle::operator()(const Point& point)
{
  PetscReal x = point.x() - geom_.center[X];
  PetscReal y = point.y() - geom_.center[Y];
  return (x * x + y * y) > POW2(geom_.radius);
}
