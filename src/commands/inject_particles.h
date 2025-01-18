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

  using MomentumGenerator =
    std::function<Vector3R(const Vector3R& /* reference */)>;

  InjectParticles( //
    interfaces::Particles& ionized,                 //
    interfaces::Particles& ejected,                 //
    timestep_t injection_start,                     //
    timestep_t injection_end,                       //
    PetscInt per_step_particles_num,                //
    const CoordinateGenerator& generate_coordinate, //
    const MomentumGenerator& generate_vi,           //
    const MomentumGenerator& generate_ve);

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

  std::string get_ionized_name() const;
  std::string get_ejected_name() const;
  PetscReal get_ionized_energy() const;
  PetscReal get_ejected_energy() const;

private:
  using Particles = interfaces::Particles;
  Particles& ionized_;
  Particles& ejected_;

  timestep_t injection_start_;
  timestep_t injection_end_;
  PetscInt per_step_particles_num_;

  CoordinateGenerator generate_coordinate_;
  MomentumGenerator generate_vi_;
  MomentumGenerator generate_ve_;

  PetscReal energy_i_ = 0.0;
  PetscReal energy_e_ = 0.0;
};

#endif // SRC_COMMANDS_INJECT_PARTICLES_H
