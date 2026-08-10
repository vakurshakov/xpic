#include "src/algorithms/adaptive_substepping.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

struct State {
  PetscReal x = 0.0;
};

void require(bool condition, const char* message)
{
  if (!condition)
    throw std::runtime_error(message);
}

auto midpoint = [](const State& start, const State& end) {
  return State{0.5 * (start.x + end.x)};
};

void test_parent_retry_then_converged_halves()
{
  drift_kinetic::AdaptiveSubstepStats stats;
  State end{};
  PetscInt accepted = 0;
  PetscReal accepted_dt = 0.0;

  auto attempt = [](PetscReal dt, State& next, const State& start) {
    next.x = start.x + dt;
    return drift_kinetic::AdaptiveSubstepAttempt{
      dt <= 0.5, 3, dt, 2.0 * dt};
  };
  auto accept = [&](PetscReal dt, State&, const State&) {
    ++accepted;
    accepted_dt += dt;
  };

  drift_kinetic::adaptive_substep(
    1.0, end, State{}, 4, attempt, midpoint, accept, stats);

  require(stats.retries == 1,
    "failed parent attempt was not counted as one retry");
  require(stats.leaf_failures == 0,
    "converged half steps were incorrectly counted as leaf failures");
  require(stats.iteration_sum == 9 && stats.max_iterations == 3,
    "iteration statistics do not include all nonlinear attempts");
  require(accepted == 2 && std::abs(accepted_dt - 1.0) < 1e-15,
    "only the two accepted half steps must be deposited");
  require(drift_kinetic::adaptive_substep_status(stats, true) == PETSC_SUCCESS,
    "strict mode rejected a trajectory whose leaves converged");
}

void test_terminal_leaf_failure_is_strict_error()
{
  drift_kinetic::AdaptiveSubstepStats stats;
  State end{};
  PetscInt accepted = 0;

  auto attempt = [](PetscReal dt, State& next, const State& start) {
    next.x = start.x + dt;
    return drift_kinetic::AdaptiveSubstepAttempt{
      false, 60, dt, 10.0 * dt};
  };
  auto accept = [&](PetscReal, State&, const State&) { ++accepted; };

  drift_kinetic::adaptive_substep(
    1.0, end, State{}, 2, attempt, midpoint, accept, stats);

  require(stats.retries == 3,
    "each failed non-leaf attempt must be counted as a retry");
  require(stats.leaf_failures == 4,
    "all failed leaves at maximum depth must be counted");
  require(accepted == 4,
    "terminal leaves must remain accepted in compatibility mode");
  require(std::abs(stats.max_leaf_residue_r - 0.25) < 1e-15 &&
      std::abs(stats.max_leaf_residue_v - 2.5) < 1e-15,
    "maximum terminal residuals are incorrect");
  require(drift_kinetic::adaptive_substep_status(stats, false) == PETSC_SUCCESS,
    "compatibility mode must preserve non-fatal terminal failures");
  require(drift_kinetic::adaptive_substep_status(stats, true) ==
      PETSC_ERR_NOT_CONVERGED,
    "strict mode did not return PETSC_ERR_NOT_CONVERGED");
}

void test_nonfinite_terminal_residue_is_visible()
{
  drift_kinetic::AdaptiveSubstepStats stats;
  State end{};
  auto attempt = [](PetscReal, State&, const State&) {
    return drift_kinetic::AdaptiveSubstepAttempt{
      false, 60, std::numeric_limits<PetscReal>::quiet_NaN(),
      std::numeric_limits<PetscReal>::infinity()};
  };
  auto accept = [](PetscReal, State&, const State&) {};

  drift_kinetic::adaptive_substep(
    1.0, end, State{}, 0, attempt, midpoint, accept, stats);

  require(stats.leaf_failures == 1 &&
      stats.max_leaf_residue_r == PETSC_MAX_REAL &&
      stats.max_leaf_residue_v == PETSC_MAX_REAL,
    "non-finite terminal residual was hidden by max aggregation");
}

}  // namespace

int main()
{
  try {
    test_parent_retry_then_converged_halves();
    test_terminal_leaf_failure_is_strict_error();
    test_nonfinite_terminal_residue_is_visible();
  }
  catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
