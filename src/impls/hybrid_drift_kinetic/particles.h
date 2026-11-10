#ifndef SRC_IMPLS_HYBRID_DRIFT_KINETIC_PARTICLES_H
#define SRC_IMPLS_HYBRID_DRIFT_KINETIC_PARTICLES_H

#include "src/interfaces/particles.h"

namespace hybrid_drift_kinetic {

/// @brief Fully kinetic sort, pushed by the implicit Crank-Nicolson scheme
/// of `eccapfim` with the energy-conserving Esirkepov interpolation.
class KineticParticles : public interfaces::Particles {
public:
  KineticParticles(const World& world, const SortParameters& parameters);

  PetscErrorCode initialize() override;
  PetscErrorCode prepare_storage();
  PetscErrorCode form_iteration();

  PetscReal kinetic_energy_local() const;
  PetscReal get_average_iteration_number() const;

private:
  PetscInt size = 0;    // Local particle count.
  PetscReal avgit = 0;  // Average pusher iteration count.
};

}  // namespace hybrid_drift_kinetic

#endif  // SRC_IMPLS_HYBRID_DRIFT_KINETIC_PARTICLES_H
