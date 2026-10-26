#include "set_particles.h"

#include "src/diagnostics/energy.h"

SetParticles::SetParticles(std::vector<Sort> sorts,
  PetscInt number_of_particles, const CoordinateGenerator& generate_coordinate)
  : sorts(sorts),
    generate_coordinate(generate_coordinate),
    added_energy(sorts.size()),
    added_particles(sorts.size()),
    number_of_particles(number_of_particles)
{
}

PetscErrorCode SetParticles::execute(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  PetscCall(load(number_of_particles));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetParticles::load(PetscInt number_of_particles)
{
  PetscFunctionBeginUser;
  std::ranges::fill(added_energy, 0);
  std::ranges::fill(added_particles, 0);

  // The loop is sequential: coordinate generators can be stateful.
  for (PetscInt p = 0; p < number_of_particles; p++) {
    Vector3R coordinate = generate_coordinate();

    for (std::size_t s = 0; s < sorts.size(); s++) {
      auto& [particles, generate_momentum] = sorts[s];
      Vector3R momentum = generate_momentum(coordinate);

      bool is_added = false;
      PetscCall(particles.add_particle(Point(coordinate, momentum), &is_added));

      if (is_added) {
        const PetscReal m = particles.parameters.m;
        const PetscReal mpw = particles.parameters.mpw;

        added_energy[s] += Energy::get_kinetic(momentum, m, mpw);
        added_particles[s]++;
      }
    }
  }

  for (auto& sort : sorts)
    PetscCall(sort.particles.commit_added_particles());

  PetscCall(log_statistics());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetParticles::log_statistics()
{
  PetscFunctionBeginUser;
  const PetscInt n = static_cast<PetscInt>(sorts.size());
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, added_energy.data(), n, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, added_particles.data(), n, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

  for (std::size_t s = 0; s < sorts.size(); s++) {
    auto& particles = sorts[s].particles;
    LOG("  Particles have been added into \"{}\"", particles.parameters.sort_name);
    PetscCall(MPIUtils::log_statistics("    ", added_particles[s], PETSC_COMM_WORLD));
    LOG("    energy: {:6.4e}", added_energy[s]);
    PetscCall(particles.log_distribution());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
