#include "remove_particles.h"

#include "src/diagnostics/particles_energy.h"
#include "src/utils/configuration.h"

RemoveParticles::RemoveParticles(interfaces::Particles& particles, Tester&& test)
  : particles_(particles), within_geom_(std::move(test))
{
}

PetscErrorCode RemoveParticles::execute(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  PetscInt removed_particles = 0;
  removed_energy_ = 0.0;

  const World& w = particles_.world;
  const PetscInt Np = particles_.parameters.Np;
  const PetscReal m = particles_.parameters.m;

  for (PetscInt g = 0; g < w.size.elements_product(); ++g) {
    const Vector3R r{
      (g % w.size[X]) * dx,
      ((g / w.size[X]) % w.size[Y]) * dy,
      ((g / w.size[X]) / w.size[Y]) * dz,
    };

    auto& cell = particles_.storage[g];
    if (!cell.empty() && !within_geom_(r)) {
      removed_particles += cell.size();

      for (const auto& [_, p] : cell)
        removed_energy_ += ParticlesEnergy::get(p, m, Np);

      cell.clear();
    }
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
