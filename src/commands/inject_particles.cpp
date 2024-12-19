#include "inject_particles.h"

#include "src/utils/configuration.h"
#include "src/utils/random_generator.h"

InjectParticles::InjectParticles(interfaces::Particles& ionized,
  interfaces::Particles& ejected, timestep_t injection_start,
  timestep_t injection_end, PetscInt per_step_particles_num,
  const Coordinate_generator& set_point_of_birth,
  const Velocity_generator& load_momentum_i,
  const Velocity_generator& load_momentum_e)
  : ionized_(ionized),
    ejected_(ejected),
    injection_start_(injection_start),
    injection_end_(injection_end),
    per_step_particles_num_(per_step_particles_num),
    generate_coordinate_(set_point_of_birth),
    generate_vi_(load_momentum_i),
    generate_ve_(load_momentum_e)
{
  // ionized_.points_.reserve(per_step_particles_num_ * (injection_end_ - injection_start_) + 10'000);
  // ejected_.points_.reserve(per_step_particles_num_ * (injection_end_ - injection_start_) + 10'000);
}

PetscErrorCode InjectParticles::execute(timestep_t t)
{
  PetscFunctionBeginUser;
  const PetscInt Npi = ionized_.parameters().Np;
  const PetscReal mi = ionized_.parameters().m;

  const PetscInt Npe = ejected_.parameters().Np;
  const PetscReal me = ejected_.parameters().m;

  if (t < injection_start_)
    PetscFunctionReturn(PETSC_SUCCESS);

  PetscReal loaded_energy_i = 0;
  PetscReal loaded_energy_e = 0;

  for (PetscInt p = 0; p < per_step_particles_num_; ++p) {
    Vector3R shared_coordinate = generate_coordinate_();
    Vector3R vi = generate_vi_(shared_coordinate);
    Vector3R ve = generate_ve_(shared_coordinate);

    loaded_energy_i += 0.5 * (mi * vi.squared()) * dx * dy / Npi;
    loaded_energy_e += 0.5 * (me * ve.squared()) * dx * dy / Npe;

    ionized_.add_particle(Point(shared_coordinate, vi));
    ejected_.add_particle(Point(shared_coordinate, ve));
  }

  LOG("Ionized {} energy per step = {}", ionized_.parameters().sort_name, loaded_energy_i);
  LOG("Ejected {} energy per step = {}", ejected_.parameters().sort_name, loaded_energy_e);
  PetscFunctionReturn(PETSC_SUCCESS);
}
