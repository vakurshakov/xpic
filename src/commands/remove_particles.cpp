#include "remove_particles.h"

#include "src/diagnostics/particles_energy.h"
#include "src/utils/configuration.h"
#include "src/utils/region_operations.h"

RemoveParticles::RemoveParticles(interfaces::Particles& particles, Tester&& test)
  : particles_(particles), should_remove_(std::move(test))
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
        removed_energy_ += ParticlesEnergy::get(point.p, m, Np);
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


bool RemoveFromBox::operator()(const Point& point)
{
  return  //
    (geom_.min[X] > point.x() || point.x() > geom_.max[X]) ||
    (geom_.min[Y] > point.y() || point.y() > geom_.max[Y]) ||
    (geom_.min[Z] > point.z() || point.z() > geom_.max[Z]);
}


bool RemoveFromCylinder::operator()(const Point& point)
{
  PetscReal x = point.x() - geom_.center[X];
  PetscReal y = point.y() - geom_.center[Y];
  PetscReal z = point.z() - geom_.center[Z];
  return std::abs(z) > 0.5 * geom_.height || (x * x + y * y) > POW2(geom_.radius);
}
