#ifndef SRC_COMMANDS_SET_PARTICLES_H
#define SRC_COMMANDS_SET_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/particles.h"
#include "src/utils/particles_load.hpp"

class SetParticles : public interfaces::Command {
public:
  SetParticles( //
    interfaces::Particles& particles,               //
    PetscInt number_of_particles,                   //
    const CoordinateGenerator& generate_coordinate, //
    const MomentumGenerator& generate_momentum);

  PetscErrorCode execute(PetscInt t) override;

private:
  PetscErrorCode log_statistics();

  using Particles = interfaces::Particles;
  Particles& particles_;

  PetscInt number_of_particles_;

  CoordinateGenerator generate_coordinate_;
  MomentumGenerator generate_momentum_;

  PetscReal energy_ = 0.0;
  PetscInt added_particles_ = 0;
};

#endif // SRC_COMMANDS_SET_PARTICLES_H
