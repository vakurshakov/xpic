#include "set_particles.h"

#include "src/diagnostics/particles_energy.h"
#include "src/utils/configuration.h"
#include "src/utils/random_generator.h"

SetParticles::SetParticles(                       //
  interfaces::Particles& particles,               //
  PetscInt number_of_particles,                   //
  const CoordinateGenerator& generate_coordinate, //
  const MomentumGenerator& generate_momentum)
  : particles_(particles),
    number_of_particles_(number_of_particles),
    generate_coordinate_(generate_coordinate),
    generate_momentum_(generate_momentum)
{
}

PetscErrorCode SetParticles::execute(PetscInt /* t */)
{
  energy_ = 0.0;
  added_particles_ = 0;

  const PetscReal m = particles_.parameters.m;
  const PetscInt Np = particles_.parameters.Np;

  PetscFunctionBeginUser;
  for (PetscInt p = 0; p < number_of_particles_; ++p) {
    Vector3R coordinate = generate_coordinate_();
    Vector3R momentum = generate_momentum_(coordinate);

    bool is_added = false;
    particles_.add_particle(Point(coordinate, momentum), &is_added);

    if (is_added) {
      energy_ += ParticlesEnergy::get(momentum, m, Np);
      added_particles_++;
    }
  }

  PetscCall(log_statistics());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetParticles::log_statistics()
{
  PetscFunctionBeginUser;
  LOG("  Particles have been added into \"{}\"", particles_.parameters.sort_name);

  PetscInt tot, min, max;
  PetscCallMPI(MPI_Allreduce(&added_particles_, &tot, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&added_particles_, &min, 1, MPIU_INT, MPI_MIN, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&added_particles_, &max, 1, MPIU_INT, MPI_MAX, PETSC_COMM_WORLD));
  added_particles_ = tot;

  PetscReal rat = min > 0 ? (PetscReal)max / min : -1.0;
  LOG("    total: {}, min: {}, max: {}, ratio: {:4.3f}", tot, min, max, rat);

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &energy_, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  LOG("    energy: {:6.4e}", energy_);
  PetscFunctionReturn(PETSC_SUCCESS);
}
