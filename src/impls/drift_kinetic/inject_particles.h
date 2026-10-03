#ifndef SRC_IMPLS_DRIFT_KINETIC_INJECT_PARTICLES_H
#define SRC_IMPLS_DRIFT_KINETIC_INJECT_PARTICLES_H

#include "src/pch.h"
#include "src/impls/drift_kinetic/particles.h"
#include "src/interfaces/command.h"
#include "src/utils/particles_load.h"

namespace drift_kinetic {

class DriftKineticEsirkepov;

class InjectParticles : public interfaces::Command {
public:
  InjectParticles(                                   //
    Particles& ionized,                              //
    Particles& ejected,                              //
    PetscInt injection_start,                        //
    PetscInt injection_end,                          //
    PetscInt per_step_particles_num,                 //
    const CoordinateGenerator& generate_coordinate,  //
    const MomentumGenerator& generate_momentum_i,    //
    const MomentumGenerator& generate_momentum_e);

  PetscErrorCode execute(PetscInt t) override;

private:
  PetscErrorCode log_statistics(PetscInt added_particles,
    PetscReal energy_i, PetscReal energy_e) const;

  PetscErrorCode add_particle(Particles& particles,
    DriftKineticEsirkepov& esirkepov, const Point& point, bool& is_added);

  Particles& ionized_;
  Particles& ejected_;

  PetscInt injection_start_;
  PetscInt injection_end_;

  /// @todo Number of injected particles can vary, if `MPI_Comm_size() > 1`
  PetscInt per_step_particles_num_;

  CoordinateGenerator generate_coordinate_;
  MomentumGenerator generate_momentum_i_;
  MomentumGenerator generate_momentum_e_;
};

}  // namespace drift_kinetic

#endif  // SRC_IMPLS_DRIFT_KINETIC_INJECT_PARTICLES_H
