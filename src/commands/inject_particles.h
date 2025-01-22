#ifndef SRC_COMMANDS_INJECT_PARTICLES_H
#define SRC_COMMANDS_INJECT_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/particles.h"
#include "src/utils/particles_load.hpp"

class InjectParticles : public interfaces::Command {
public:

  InjectParticles( //
    interfaces::Particles& ionized,                 //
    interfaces::Particles& ejected,                 //
    timestep_t injection_start,                     //
    timestep_t injection_end,                       //
    PetscInt per_step_particles_num,                //
    const CoordinateGenerator& generate_coordinate, //
    const MomentumGenerator& generate_momentum_i,   //
    const MomentumGenerator& generate_momentum_e);

  PetscErrorCode execute(timestep_t t) override;

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
  MomentumGenerator generate_momentum_i_;
  MomentumGenerator generate_momentum_e_;

  PetscReal energy_i_ = 0.0;
  PetscReal energy_e_ = 0.0;
};

#endif // SRC_COMMANDS_INJECT_PARTICLES_H
