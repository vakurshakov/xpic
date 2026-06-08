#ifndef SRC_IMPLS_DRIFT_KINETIC_SET_PAIRED_PARTICLES_H
#define SRC_IMPLS_DRIFT_KINETIC_SET_PAIRED_PARTICLES_H

#include "src/pch.h"
#include "src/algorithms/implicit_drift_kinetic.h"
#include "src/impls/drift_kinetic/particles.h"
#include "src/interfaces/command.h"
#include "src/utils/particles_load.h"

namespace drift_kinetic {

/// @brief Loads two DK sorts together from a single shared coordinate stream.
/// For each particle index `p`, one `r = generate_coordinate_()` is drawn
/// and used as the position of both species, so the global RNG cannot drift
/// them apart. Each species still gets its own momentum sample.
///
/// Intended pairing: ions + electrons placed at coincident guiding centers
/// for a quasi-neutral start. Combined with `Particles::coord_is_gc(true)`,
/// this guarantees `||r_e_gc - r_i_gc|| == 0` after loading.
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
    DriftKineticEsirkepov& esirkepov, const Point& point, bool* is_added);

  PetscErrorCode log_statistics();

  Particles& ionized_;
  Particles& ejected_;

  PetscInt number_of_particles_;

  CoordinateGenerator generate_coordinate_;
  MomentumGenerator generate_momentum_i_;
  MomentumGenerator generate_momentum_e_;

  PetscReal energy_i_ = 0.0;
  PetscReal energy_e_ = 0.0;
  PetscInt added_particles_ = 0;

  bool executed_ = false;
};

}  // namespace drift_kinetic

#endif  // SRC_IMPLS_DRIFT_KINETIC_SET_PAIRED_PARTICLES_H
