#include "inject_particles.h"

InjectParticles::InjectParticles(   //
  std::vector<Sort> sorts,          //
  PetscInt injection_start,         //
  PetscInt injection_end,           //
  PetscInt per_step_particles_num,  //
  const CoordinateGenerator& generate_coordinate)
  : SetParticles(std::move(sorts), per_step_particles_num, generate_coordinate),
    injection_start(injection_start),
    injection_end(injection_end)
{
}

PetscErrorCode InjectParticles::execute(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t < injection_start || t > injection_end) {
    std::ranges::fill(added_energy, 0);
    std::ranges::fill(added_particles, 0);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(SetParticles::execute(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

std::string InjectParticles::get_ionized_name() const
{
  return sorts[0].particles.parameters.sort_name;
}

std::string InjectParticles::get_ejected_name() const
{
  return sorts[1].particles.parameters.sort_name;
}

PetscReal InjectParticles::get_ionized_energy() const
{
  return added_energy[0];
}

PetscReal InjectParticles::get_ejected_energy() const
{
  return added_energy[1];
}
