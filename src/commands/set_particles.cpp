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
  PetscFunctionBeginUser;
  PetscReal energy = 0.0;

  const PetscInt Np = particles_.parameters.Np;
  const PetscReal m = particles_.parameters.m;

  for (PetscInt p = 0; p < number_of_particles_; ++p) {
    Vector3R coordinate = generate_coordinate_();
    Vector3R momentum = generate_momentum_(coordinate);

    energy += ParticlesEnergy::get(momentum, m, Np);
    particles_.add_particle(Point(coordinate, momentum));
  }

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));

  constexpr auto message =
    "  Particles are added into \"{}\"; particles: {}, energy: {}";
  LOG(message, particles_.parameters.sort_name, number_of_particles_, energy);
  PetscFunctionReturn(PETSC_SUCCESS);
}
