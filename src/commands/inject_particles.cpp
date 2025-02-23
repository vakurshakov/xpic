#include "inject_particles.h"

#include "src/utils/configuration.h"
#include "src/utils/random_generator.h"

InjectParticles::InjectParticles(                  //
  interfaces::Particles& ionized,                  //
  interfaces::Particles& ejected,                  //
  PetscInt injection_start,                      //
  PetscInt injection_end,                        //
  PetscInt per_step_particles_num,                 //
  const CoordinateGenerator& generate_coordinate,  //
  const MomentumGenerator& generate_momentum_i,    //
  const MomentumGenerator& generate_momentum_e)
  : ionized_(ionized),
    ejected_(ejected),
    injection_start_(injection_start),
    injection_end_(injection_end),
    per_step_particles_num_(per_step_particles_num),
    generate_coordinate_(generate_coordinate),
    generate_momentum_i_(generate_momentum_i),
    generate_momentum_e_(generate_momentum_e)
{
}

PetscErrorCode InjectParticles::execute(PetscInt t)
{
  energy_i_ = 0.0;
  energy_e_ = 0.0;

  if (t < injection_start_ || t > injection_end_)
    return PETSC_SUCCESS;

  PetscFunctionBeginUser;
  for (PetscInt p = 0; p < per_step_particles_num_; ++p) {
    Vector3R shared_coordinate = generate_coordinate_();
    Vector3R pi = generate_momentum_i_(shared_coordinate);
    Vector3R pe = generate_momentum_e_(shared_coordinate);

    ionized_.add_particle(Point(shared_coordinate, pi), &energy_i_);
    ejected_.add_particle(Point(shared_coordinate, pe), &energy_e_);
  }

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &energy_i_, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &energy_e_, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));

  constexpr auto message =
    "  Particles are added into \"{}\"; particles: {}, energy: {}";
  LOG(message, ionized_.parameters.sort_name, per_step_particles_num_, energy_i_);
  LOG(message, ejected_.parameters.sort_name, per_step_particles_num_, energy_e_);
  PetscFunctionReturn(PETSC_SUCCESS);
}

std::string InjectParticles::get_ionized_name() const
{
  return ionized_.parameters.sort_name;
}

std::string InjectParticles::get_ejected_name() const
{
  return ejected_.parameters.sort_name;
}

PetscReal InjectParticles::get_ionized_energy() const
{
  return energy_i_;
}

PetscReal InjectParticles::get_ejected_energy() const
{
  return energy_e_;
}
