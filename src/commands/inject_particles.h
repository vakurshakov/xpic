#ifndef SRC_COMMANDS_INJECT_PARTICLES_H
#define SRC_COMMANDS_INJECT_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/particles.h"

/**
 * @brief Sets the fixed number of particles
 * into the computational domain each time step.
 */
class Inject_particles : public interfaces::Command {
public:
  using Coordinate_generator = std::function<Vector3R()>;
  using Velocity_generator = std::function<Vector3R(const Vector3R&)>;

  Inject_particles(interfaces::Particles& ionized,
    interfaces::Particles& ejected, timestep_t injection_start,
    timestep_t injection_end, PetscInt per_step_particles_num,
    const Coordinate_generator& set_point_of_birth,
    const Velocity_generator& load_momentum_i,
    const Velocity_generator& load_momentum_e);

  /**
   * @brief Loads the number of particles (per_step_particles_num)
   * in pairs into a computational domain with some space and
   * momentum distributions.
   *
   * @param t Outer time step to start from injection_start.
   */
  PetscErrorCode execute(timestep_t t) override;

  bool needs_to_be_removed(timestep_t t) const override
  {
    return t >= injection_end_;
  }

private:
  using Particles = interfaces::Particles;
  Particles& ionized_;
  Particles& ejected_;

  timestep_t injection_start_;
  timestep_t injection_end_;
  PetscInt per_step_particles_num_;

  Coordinate_generator generate_coordinate_;
  Velocity_generator generate_vi_;
  Velocity_generator generate_ve_;
};

#endif // SRC_COMMANDS_INJECT_PARTICLES_H
