#ifndef SRC_ALGORITHMS_ADAPTIVE_SUBSTEPPING_H
#define SRC_ALGORITHMS_ADAPTIVE_SUBSTEPPING_H

#include <algorithm>
#include <cmath>

#include <petscsys.h>

namespace drift_kinetic {

/// Result of one nonlinear attempt on an adaptive particle substep.
struct AdaptiveSubstepAttempt {
  bool converged = false;
  PetscInt iterations = 0;
  PetscReal residue_r = 0.0;
  PetscReal residue_v = 0.0;
};

/// Statistics accumulated over the complete adaptive substep tree.
struct AdaptiveSubstepStats {
  PetscInt retries = 0;
  PetscInt leaf_failures = 0;
  PetscInt iteration_sum = 0;
  PetscInt max_iterations = 0;
  PetscReal max_leaf_residue_r = 0.0;
  PetscReal max_leaf_residue_v = 0.0;
};

namespace detail {

template<class State, class Attempt, class Midpoint, class Accept>
void adaptive_substep_impl(PetscReal dt, State& end, const State& start,
  PetscInt depth, PetscInt max_depth, Attempt& attempt, Midpoint& midpoint,
  Accept& accept, AdaptiveSubstepStats& stats)
{
  const AdaptiveSubstepAttempt result = attempt(dt, end, start);
  stats.iteration_sum += result.iterations;
  stats.max_iterations = std::max(stats.max_iterations, result.iterations);

  if (!result.converged && depth < max_depth) {
    ++stats.retries;
    State mid = midpoint(start, end);
    const PetscReal half_dt = 0.5 * dt;
    adaptive_substep_impl(half_dt, mid, start, depth + 1, max_depth,
      attempt, midpoint, accept, stats);
    adaptive_substep_impl(half_dt, end, mid, depth + 1, max_depth,
      attempt, midpoint, accept, stats);
    return;
  }

  if (!result.converged) {
    ++stats.leaf_failures;
    const PetscReal residue_r = std::isfinite(result.residue_r)
      ? result.residue_r : PETSC_MAX_REAL;
    const PetscReal residue_v = std::isfinite(result.residue_v)
      ? result.residue_v : PETSC_MAX_REAL;
    stats.max_leaf_residue_r =
      std::max(stats.max_leaf_residue_r, residue_r);
    stats.max_leaf_residue_v =
      std::max(stats.max_leaf_residue_v, residue_v);
  }

  // A leaf is the only attempt contributing to the accepted trajectory and
  // current deposition. Parent attempts that caused a retry are discarded.
  accept(dt, end, start);
}

}  // namespace detail

/// Runs one adaptive step. Failed parent attempts are split in half until a
/// converged attempt or `max_depth` is reached.
template<class State, class Attempt, class Midpoint, class Accept>
void adaptive_substep(PetscReal dt, State& end, const State& start,
  PetscInt max_depth, Attempt&& attempt, Midpoint&& midpoint, Accept&& accept,
  AdaptiveSubstepStats& stats)
{
  detail::adaptive_substep_impl(dt, end, start, 0, max_depth,
    attempt, midpoint, accept, stats);
}

/// Converts terminal leaf failures into an error only in strict mode.
inline PetscErrorCode adaptive_substep_status(
  const AdaptiveSubstepStats& stats, bool fail_on_terminal_nonconvergence)
{
  return fail_on_terminal_nonconvergence && stats.leaf_failures > 0
    ? PETSC_ERR_NOT_CONVERGED
    : PETSC_SUCCESS;
}

}  // namespace drift_kinetic

#endif  // SRC_ALGORITHMS_ADAPTIVE_SUBSTEPPING_H
