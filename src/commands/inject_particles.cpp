#include "inject_particles.h"

#include "src/diagnostics/particles_energy.h"
#include "src/utils/configuration.h"
#include "src/utils/random_generator.h"

InjectParticles::InjectParticles(                  //
  interfaces::Particles& ionized,                  //
  interfaces::Particles& ejected,                  //
  PetscInt injection_start,                        //
  PetscInt injection_end,                          //
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
  added_particles_ = 0;

  if (t < injection_start_ || t > injection_end_)
    return PETSC_SUCCESS;

  const PetscReal mi = ionized_.parameters.m;
  const PetscInt Npi = ionized_.parameters.Np;

  const PetscReal me = ejected_.parameters.m;
  const PetscInt Npe = ejected_.parameters.Np;

  PetscFunctionBeginUser;
  for (PetscInt p = 0; p < per_step_particles_num_; ++p) {
    Vector3R shared_coordinate = generate_coordinate_();
    Vector3R pi = generate_momentum_i_(shared_coordinate);
    Vector3R pe = generate_momentum_e_(shared_coordinate);

    bool is_added = false;
    ionized_.add_particle(Point(shared_coordinate, pi), &is_added);
    ejected_.add_particle(Point(shared_coordinate, pe), &is_added);

    if (is_added) {
      energy_i_ += ParticlesEnergy::get(pi, mi, Npi);
      energy_e_ += ParticlesEnergy::get(pe, me, Npe);
      added_particles_++;
    }
  }

  PetscCall(log_statistics());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode InjectParticles::log_statistics()
{
  PetscFunctionBeginUser;
  LOG("  Particles have been injected");

  PetscCall(MPIUtils::log_statistics("    ", added_particles_, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &added_particles_, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

  const std::vector<std::pair<std::string, PetscReal&>> map{
    {ionized_.parameters.sort_name, energy_i_},
    {ejected_.parameters.sort_name, energy_e_},
  };

  for (auto&& [name, energy] : map) {
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
    LOG("    energy added into \"{}\": {:6.4e}", name, energy);
  }
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
