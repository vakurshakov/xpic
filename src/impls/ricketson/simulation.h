#ifndef SRC_RICKETSON_SIMULATION_H
#define SRC_RICKETSON_SIMULATION_H

#include "src/interfaces/simulation.h"

#include <petscdmda.h>

#include "src/impls/ricketson/particles.h"

namespace ricketson {

class Simulation : public interfaces::Simulation {
public:
  Simulation() = default;
  ~Simulation() override;

  /// @note α ∈ (0, 1), controls how close to the time-step restrictions one is willing to get.
  const PetscReal alpha = 0.9;

  /// @note β ∈ (0, 1), controls the region |v_E / u{n+1/2} - 1| < β, where conserving effective force becomes discontinuous.
  const PetscReal beta = 0.2;

  /// @note ε ∈ (0, 1), controls the maximum permissible fractional change in μ within a time-step.
  const PetscReal epsilon = 0.15;

  /// @note Γ > 0, measures the accuracy with which we wish to resolve spatial variations in the magnetic field.
  const PetscReal gamma = 0.1;

  /// @note The smallest timescale in the problem that we wish to resolve.
  const PetscReal t_res = 0.1;

  DM da_;
  Vec E_;
  Vec B_;
  Vec B_grad_;
  Vec B_norm_;

  DMBoundaryType bounds_[3];
  std::vector<Particles> particles_;

  PetscErrorCode calculate_B_norm_gradient();

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(timestep_t timestep) override;

  PetscErrorCode setup_norm_gradient();
  PetscInt index(PetscInt k, PetscInt j, PetscInt i);
  PetscInt index(PetscInt k, PetscInt j, PetscInt i, PetscInt l);

  Mat norm_gradient_;
};

}

#endif // SRC_RICKETSON_SIMULATION_H
