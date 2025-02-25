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
  removed_energy_ = 0.0;
  removed_particles_ = 0;

  const World& world = particles_.world;

  const PetscReal m = particles_.parameters.m;
  const PetscInt Np = particles_.parameters.Np;

  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    auto& cell = particles_.storage[g];

    const Vector3R r{
      world.start[X] + (g % world.size[X]) * dx,
      world.start[Y] + ((g / world.size[X]) % world.size[Y]) * dy,
      world.start[Z] + ((g / world.size[X]) / world.size[Y]) * dz,
    };

    if (cell.empty() || within_geom_(r))
      continue;

    for (const auto& [_, p] : cell)
      removed_energy_ += ParticlesEnergy::get(p, m, Np);

    removed_particles_ += cell.size();
    cell.clear();
  }

  PetscCall(log_statistics());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RemoveParticles::log_statistics()
{
  PetscFunctionBeginUser;
  LOG("  Particles have been removed from \"{}\"", particles_.parameters.sort_name);
  PetscCall(MPIUtils::log_statistics("    ", removed_particles_, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &removed_energy_, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &removed_particles_, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  LOG("    energy: {:6.4e}", removed_energy_);
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
