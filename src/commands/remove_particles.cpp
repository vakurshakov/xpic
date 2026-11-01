#include "remove_particles.h"

#include "src/diagnostics/energy.h"
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

  const PetscReal m = particles_.parameters.m;
  const PetscReal mpw = particles_.parameters.n / particles_.parameters.Np;

  // The geometry is tested at the corner of the cell of the particle
  auto is_removed = [&](const Point& point)
  {
    const Vector3R r{
      FLOOR_STEP(point.x(), dx) * dx,
      FLOOR_STEP(point.y(), dy) * dy,
      FLOOR_STEP(point.z(), dz) * dz,
    };
    return !within_geom_(r);
  };

  std::vector<Point> removed;
  PetscCall(particles_.extract(is_removed, &removed));

  for (const Point& point : removed)
    removed_energy_ += Energy::get_kinetic(point.p, m, mpw);
  removed_particles_ = (PetscInt)removed.size();

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
