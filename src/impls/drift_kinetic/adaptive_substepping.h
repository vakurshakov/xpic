#ifndef SRC_IMPLS_DRIFT_KINETIC_ADAPTIVE_SUBSTEPPING_H
#define SRC_IMPLS_DRIFT_KINETIC_ADAPTIVE_SUBSTEPPING_H

#include "src/pch.h"

namespace drift_kinetic {

constexpr PetscInt max_depth = 4;

/// @brief Result of a particle substep attempt.
struct AdaptiveSubstepAttempt {
  bool converged = false;
  PetscInt iterations = 0;
  PetscReal residue_r = 0;
  PetscReal residue_v = 0;
};

/// @brief Iteration and failure statistics for an adaptive step.
struct AdaptiveSubstepStats {
  PetscInt retries = 0;
  PetscInt leaf_failures = 0;
  PetscInt iteration_sum = 0;
  PetscInt max_iterations = 0;
  PetscReal max_leaf_residue_r = 0;
  PetscReal max_leaf_residue_v = 0;
};

/// @brief Splits failed substeps recursively and accepts only terminal attempts.
template<class State, class Attempt, class Midpoint, class Accept>
void adaptive_substep_impl(PetscReal dt, State& end, const State& start,
  PetscInt depth, Attempt& attempt, Midpoint& midpoint, Accept& accept,
  AdaptiveSubstepStats& stats)
{
  const AdaptiveSubstepAttempt result = attempt(dt, end, start);

  stats.iteration_sum += result.iterations;
  stats.max_iterations = std::max(stats.max_iterations, result.iterations);

  if (!result.converged && depth < max_depth) {
    stats.retries++;
    State mid = midpoint(start, end);

    adaptive_substep_impl(0.5 * dt, mid, start, depth + 1,  //
      attempt, midpoint, accept, stats);

    adaptive_substep_impl(0.5 * dt, end, mid, depth + 1,  //
      attempt, midpoint, accept, stats);
    return;
  }

  if (!result.converged) {
    stats.leaf_failures++;

    stats.max_leaf_residue_r = std::max(stats.max_leaf_residue_r,
      isfinite(result.residue_r) ? result.residue_r : PETSC_MAX_REAL);

    stats.max_leaf_residue_v = std::max(stats.max_leaf_residue_v,
      isfinite(result.residue_v) ? result.residue_v : PETSC_MAX_REAL);
  }

  accept(dt, end, start);
}

/// @brief Advances a step, halving failed attempts up to the depth limit.
template<class State, class Attempt, class Midpoint, class Accept>
void adaptive_substep(PetscReal dt, State& end, const State& start,
  Attempt&& attempt, Midpoint&& midpoint, Accept&& accept,
  AdaptiveSubstepStats& stats)
{
  adaptive_substep_impl(dt, end, start, 0, attempt, midpoint, accept, stats);
}

}  // namespace drift_kinetic

#endif  // SRC_IMPLS_DRIFT_KINETIC_ADAPTIVE_SUBSTEPPING_H
