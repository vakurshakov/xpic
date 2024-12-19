#ifndef SRC_COMMANDS_INJECT_PARTICLES_H
#define SRC_COMMANDS_INJECT_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/particles.h"

/**
 * @brief Sets the fixed number of particles
 * into the computational domain each time step.
 */
class InjectParticles : public interfaces::Command {
public:
  using CoordinateGenerator = std::function<Vector3R()>;

  using VelocityGenerator =
    std::function<Vector3R(const Vector3R& /* reference */)>;

  InjectParticles( //
    interfaces::Particles& ionized,                //
    interfaces::Particles& ejected,                //
    timestep_t injection_start,                    //
    timestep_t injection_end,                      //
    PetscInt per_step_particles_num,               //
    const CoordinateGenerator& set_point_of_birth, //
    const VelocityGenerator& load_momentum_i,      //
    const VelocityGenerator& load_momentum_e);

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

  CoordinateGenerator generate_coordinate_;
  VelocityGenerator generate_vi_;
  VelocityGenerator generate_ve_;
};

#endif // SRC_COMMANDS_INJECT_PARTICLES_H
