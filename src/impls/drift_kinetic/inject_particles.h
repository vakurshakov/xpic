#ifndef SRC_IMPLS_DRIFT_KINETIC_INJECT_PARTICLES_H
#define SRC_IMPLS_DRIFT_KINETIC_INJECT_PARTICLES_H

#include "src/pch.h"
#include "src/algorithms/implicit_drift_kinetic.h"
#include "src/impls/drift_kinetic/particles.h"
#include "src/interfaces/command.h"
#include "src/utils/particles_load.h"

namespace drift_kinetic {

/// @brief Drift-kinetic analog of `::InjectParticles`. Keeps the same interface,
/// but adds the injected points directly into `Particles::dk_curr_storage`,
/// converting them to `PointByField` with the locally interpolated `B`.
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

  std::string get_ionized_name() const;
  std::string get_ejected_name() const;
  PetscReal get_ionized_energy() const;
  PetscReal get_ejected_energy() const;

private:
  PetscErrorCode log_statistics();

  /// @brief Adds a single point into `particles.dk_curr_storage`, using
  /// `esirkepov` to interpolate `B` at the point's position.
  PetscErrorCode add_particle(Particles& particles,
    DriftKineticEsirkepov& esirkepov, const Point& point, bool* is_added);

  Particles& ionized_;
  Particles& ejected_;

  PetscInt injection_start_;
  PetscInt injection_end_;

  /// @todo Number of injected particles can vary, if `MPI_Comm_size() > 1`
  PetscInt per_step_particles_num_;

  CoordinateGenerator generate_coordinate_;
  MomentumGenerator generate_momentum_i_;
  MomentumGenerator generate_momentum_e_;

  PetscReal energy_i_ = 0.0;
  PetscReal energy_e_ = 0.0;
  PetscInt added_particles_ = 0;
};

}  // namespace drift_kinetic

#endif  // SRC_IMPLS_DRIFT_KINETIC_INJECT_PARTICLES_H
