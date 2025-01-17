#include "inject_particles.h"

#include "src/utils/configuration.h"
#include "src/utils/random_generator.h"

InjectParticles::InjectParticles(                //
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
  energy_i_ = 0.0;
  energy_e_ = 0.0;

  if (t < injection_start_ || t >= injection_end_)
    return PETSC_SUCCESS;

  PetscFunctionBeginUser;
  const PetscInt Npi = ionized_.parameters().Np;
  const PetscReal mi = ionized_.parameters().m;

  const PetscInt Npe = ejected_.parameters().Np;
  const PetscReal me = ejected_.parameters().m;

  for (PetscInt p = 0; p < per_step_particles_num_; ++p) {
    Vector3R shared_coordinate = generate_coordinate_();
    Vector3R vi = generate_vi_(shared_coordinate);
    Vector3R ve = generate_ve_(shared_coordinate);

    /// @todo different formula should be used for relativity case
    energy_i_ += 0.5 * (mi * vi.squared()) * (dx * dy * dz) / Npi;
    energy_e_ += 0.5 * (me * ve.squared()) * (dx * dy * dz) / Npe;

    ionized_.add_particle(Point(shared_coordinate, vi));
    ejected_.add_particle(Point(shared_coordinate, ve));
  }

  constexpr auto message =
    "  Particles are added into \"{}\"; particles: {}, energy: {}";
  LOG(message, ionized_.parameters().sort_name, per_step_particles_num_, energy_i_);
  LOG(message, ejected_.parameters().sort_name, per_step_particles_num_, energy_e_);
  PetscFunctionReturn(PETSC_SUCCESS);
}

std::string InjectParticles::get_ionized_name() const
{
  return ionized_.parameters().sort_name;
}

std::string InjectParticles::get_ejected_name() const
{
  return ejected_.parameters().sort_name;
}

PetscReal InjectParticles::get_ionized_energy() const
{
  return energy_i_;
}

PetscReal InjectParticles::get_ejected_energy() const
{
  return energy_e_;
}
