#include "inject_particles.h"

#include "src/utils/configuration.h"
#include "src/utils/random_generator.h"

InjectParticles::InjectParticles( //
  interfaces::Particles& ionized,                //
  interfaces::Particles& ejected,                //
  timestep_t injection_start,                    //
  timestep_t injection_end,                      //
  PetscInt per_step_particles_num,               //
  const CoordinateGenerator& set_point_of_birth, //
  const VelocityGenerator& load_momentum_i,      //
  const VelocityGenerator& load_momentum_e)
  : ionized_(ionized),
    ejected_(ejected),
    injection_start_(injection_start),
    injection_end_(injection_end),
    per_step_particles_num_(per_step_particles_num),
    generate_coordinate_(set_point_of_birth),
    generate_vi_(load_momentum_i),
    generate_ve_(load_momentum_e)
{
  ionized_.reserve(per_step_particles_num_ * (injection_end_ - injection_start_));
  ejected_.reserve(per_step_particles_num_ * (injection_end_ - injection_start_));
}

PetscErrorCode InjectParticles::execute(timestep_t t)
{
  PetscFunctionBeginUser;
  if (t < injection_start_)
    PetscFunctionReturn(PETSC_SUCCESS);

  const PetscInt Npi = ionized_.parameters().Np;
  const PetscReal mi = ionized_.parameters().m;

  const PetscInt Npe = ejected_.parameters().Np;
  const PetscReal me = ejected_.parameters().m;

  PetscReal loaded_energy_i = 0;
  PetscReal loaded_energy_e = 0;

  for (PetscInt p = 0; p < per_step_particles_num_; ++p) {
    Vector3R shared_coordinate = generate_coordinate_();
    Vector3R vi = generate_vi_(shared_coordinate);
    Vector3R ve = generate_ve_(shared_coordinate);

    loaded_energy_i += 0.5 * (mi * vi.squared()) * (dx * dy * dz) / Npi;
    loaded_energy_e += 0.5 * (me * ve.squared()) * (dx * dy * dz) / Npe;

    ionized_.add_particle(Point(shared_coordinate, vi));
    ejected_.add_particle(Point(shared_coordinate, ve));
  }

  LOG("Energy of \"{}\" (ionized) added in {} step: {}", ionized_.parameters().sort_name, t, loaded_energy_i);
  LOG("Energy of \"{}\" (ejected) added in {} step: {}", ejected_.parameters().sort_name, t, loaded_energy_e);
  PetscFunctionReturn(PETSC_SUCCESS);
}
