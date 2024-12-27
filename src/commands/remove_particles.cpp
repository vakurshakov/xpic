#include "remove_particles.h"

#include "src/utils/configuration.h"

class RemoveParticles::RemoveFromCircle {
public:
  RemoveFromCircle(const CircleGeometry& geom);
  bool operator()(const Point& point);
  CircleGeometry geom_;
};


RemoveParticles::RemoveParticles(
  interfaces::Particles& particles, const CircleGeometry& geom)
  : particles_(particles), should_remove_(RemoveFromCircle(geom))
{
}

PetscErrorCode RemoveParticles::execute(timestep_t /* t */)
{
  PetscFunctionBeginUser;
  removed_energy_ = 0.0;

  const PetscInt Np = particles_.parameters().Np;
  const PetscReal m = particles_.parameters().m;

  auto& storage = particles_.points();
  auto it =
    std::remove_if(storage.begin(), storage.end(), [&](const Point& point) {
      if (should_remove_(point)) {
        removed_energy_ += 0.5 * (m * point.p.squared()) / Np * (dx * dy * dz);
        return true;
      }
      return false;
    });

  if (it != storage.end()) {
    constexpr auto message = "Particles are removed from \"{}\"; removed particles: {}, removed energy: {}";
    LOG(message, particles_.parameters().sort_name, storage.end() - it, removed_energy_);
    storage.erase(it, storage.end());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal RemoveParticles::get_removed_energy() const
{
  return removed_energy_;
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
