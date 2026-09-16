#ifndef SRC_IMPLS_DRIFT_KINETIC_SET_PAIRED_PARTICLES_H
#define SRC_IMPLS_DRIFT_KINETIC_SET_PAIRED_PARTICLES_H

#include "src/pch.h"
#include "src/impls/drift_kinetic/particles.h"
#include "src/interfaces/command.h"
#include "src/utils/particles_load.h"

namespace drift_kinetic {

class DriftKineticEsirkepov;

class SetPairedParticles : public interfaces::Command {
public:
  SetPairedParticles(                                //
    Particles& ionized,                              //
    Particles& ejected,                              //
    PetscInt number_of_particles,                    //
    const CoordinateGenerator& generate_coordinate,  //
    const MomentumGenerator& generate_momentum_i,    //
    const MomentumGenerator& generate_momentum_e);

  PetscErrorCode execute(PetscInt t) override;

private:
  PetscErrorCode add_particle(Particles& particles,
    DriftKineticEsirkepov& esirkepov, const Point& point, bool& is_added);

  PetscErrorCode log_statistics(PetscInt added_particles,
    PetscReal energy_i, PetscReal energy_e) const;

  Particles& ionized_;
  Particles& ejected_;

  PetscInt number_of_particles_;

  CoordinateGenerator generate_coordinate_;
  MomentumGenerator generate_momentum_i_;
  MomentumGenerator generate_momentum_e_;

  bool executed_ = false;
};

}  // namespace drift_kinetic

#endif  // SRC_IMPLS_DRIFT_KINETIC_SET_PAIRED_PARTICLES_H
