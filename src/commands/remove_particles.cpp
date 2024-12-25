#include "remove_particles.h"

#include "src/utils/configuration.h"

CircleGeometry::CircleGeometry(const Vector3R& center, PetscReal radius)
  : center(center), radius(radius)
{
}


RemoveParticles::RemoveParticles(
  interfaces::Particles& particles, const CircleGeometry& geom)
  : particles_(particles), remove_(RemoveFromCircle(geom))
{
}

PetscErrorCode RemoveParticles::execute(timestep_t /* t */)
{
  PetscFunctionBeginUser;
  auto& storage = particles_.points();

  auto it = std::remove_if(storage.begin(), storage.end(), remove_);
  if (it != storage.end()) {
    storage.erase(it, storage.end());
    LOG("{} particles were removed from {}", storage.end() - it, particles_.parameters().sort_name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


RemoveParticles::RemoveFromCircle::RemoveFromCircle(const CircleGeometry& geom)
  : geom_(geom)
{
}

bool RemoveParticles::RemoveFromCircle::operator()(const Point& point)
{
  PetscReal x = point.r[X] - geom_.center[X];
  PetscReal y = point.r[Y] - geom_.center[Y];
  return (x * x + y * y) > POW2(geom_.radius);
}
