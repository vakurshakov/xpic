#include "adaptive_substepping.h"

#include "src/algorithms/drift_kinetic_push.h"
#include "src/algorithms/implicit_drift_kinetic.h"

namespace drift_kinetic {

PetscReal PushStatistics::average_iterations() const
{
  return particles ? (PetscReal)iteration_sum / particles : 0;
}

void PushStatistics::merge(const PushStatistics& other)
{
  retries += other.retries;
  leaf_failures += other.leaf_failures;
  iteration_sum += other.iteration_sum;
  root_failures += other.root_failures;
  particles += other.particles;
  failed_particles += other.failed_particles;
  leaves += other.leaves;
  invalid_particles += other.invalid_particles;
  max_iterations = std::max(max_iterations, other.max_iterations);
  depth = std::max(depth, other.depth);
  max_leaf_residue_r = std::max(max_leaf_residue_r, other.max_leaf_residue_r);
  max_leaf_residue_v = std::max(max_leaf_residue_v, other.max_leaf_residue_v);
}

PetscErrorCode PushStatistics::finish(const PushStatistics& local, MPI_Comm comm,
  const std::string& sort_name, bool fail_on_any_push_nonconvergence)
{
  PetscFunctionBeginHot;
  *this = local;
  PetscInt sums[]{particles, root_failures, failed_particles, leaves,
    iteration_sum, retries, leaf_failures, invalid_particles};
  PetscInt maxima[]{depth, max_iterations};
  PetscReal residues[]{max_leaf_residue_r, max_leaf_residue_v};

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, sums, 8, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, maxima, 2, MPIU_INT, MPI_MAX, comm));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, residues, 2, MPIU_REAL, MPI_MAX, comm));
  particles = sums[0];
  root_failures = sums[1];
  failed_particles = sums[2];
  leaves = sums[3];
  iteration_sum = sums[4];
  retries = sums[5];
  leaf_failures = sums[6];
  invalid_particles = sums[7];
  depth = maxima[0];
  max_iterations = maxima[1];
  max_leaf_residue_r = residues[0];
  max_leaf_residue_v = residues[1];

  PetscCheck(invalid_particles == 0, comm, PETSC_ERR_FP,
    "Invalid drift-kinetic orbit or inaccessible stencil for %d particles", (int)invalid_particles);

  if (leaf_failures > 0 && !fail_on_any_push_nonconvergence) {
    LOG("WARNING: DK Push terminal non-convergence for sort={}:", sort_name);
    LOG("  leaf_failures={}, max FRk={: .4e}, max FVk={: .4e}", leaf_failures, max_leaf_residue_r, max_leaf_residue_v);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

AdaptiveSubstepping::AdaptiveSubstepping(PetscReal dt, PetscReal charge_weight,
  PetscReal particle_weight, PetscInt max_depth, PushStatistics& stats)
  : dt(dt),
    charge_weight(charge_weight),
    particle_weight(particle_weight),
    max_depth(max_depth),
    stats(stats)
{
}

void AdaptiveSubstepping::process(PointByField& current,
  const PointByField& previous, DriftKineticPush& push,
  DriftKineticEsirkepov& util)
{
  PushStatistics step;
  step.particles = 1;
  PointByField trial(current);
  leaves.clear();

  // Discard the entire trial before depositing any source if a leaf is invalid.
  bool valid = advance(0, 1, trial, previous, 0, push, util, step);
  if (valid) {
    for (const auto& leaf : leaves) {
      util.set_interval(leaf.begin, leaf.end);
      const PetscReal weight = leaf.end - leaf.begin;
      const Vector3R velocity = (leaf.finish.r - leaf.start.r) / (weight * dt);
      if (util.decomposition(leaf.finish.r, leaf.start.r, velocity,
            charge_weight * weight, leaf.finish.mu_p * particle_weight * weight)) {
        valid = false;
        break;
      }
    }
  }
  if (valid) {
    current = trial;
    current.p = (current.r - previous.r) / dt;
    step.leaves = (PetscInt)leaves.size();
    step.failed_particles = step.leaf_failures > 0;
  }
  else {
    step.invalid_particles = 1;
  }
  stats.merge(step);
}

bool AdaptiveSubstepping::advance(PetscReal begin, PetscReal end,
  PointByField& finish, const PointByField& start, PetscInt depth,
  DriftKineticPush& push, DriftKineticEsirkepov& util, PushStatistics& step)
{
  util.set_interval(begin, end);
  const PetscReal h = (end - begin) * dt;
  step.depth = std::max(step.depth, depth);
  if (!(h > 0) || !std::isfinite(h)) {
    if (depth == 0)
      step.root_failures++;
    return false;
  }

  const PetscErrorCode error = push.process(h, finish, start);
  const bool converged = push.has_converged();
  step.iteration_sum += push.get_iteration_number();
  step.max_iterations =
    std::max(step.max_iterations, push.get_iteration_number());
  if (depth == 0 && !converged)
    step.root_failures++;

  bool valid = error == PETSC_SUCCESS && std::isfinite(finish.p_parallel) &&
    std::isfinite(push.get_FRk()) && std::isfinite(push.get_FVk());
  for (Axis a : {X, Y, Z})
    valid = valid && std::isfinite(finish.r[a]);
  if (!valid)
    return false;

  if (!converged && depth < max_depth) {
    const PetscReal half = begin + 0.5 * (end - begin);
    if (!(begin < half && half < end))
      return false;
    step.retries++;
    PointByField mid(start);
    if (!advance(begin, half, mid, start, depth + 1, push, util, step))
      return false;
    finish = mid;
    return advance(half, end, finish, mid, depth + 1, push, util, step);
  }

  if (!converged) {
    step.leaf_failures++;
    step.max_leaf_residue_r = std::max(step.max_leaf_residue_r, push.get_FRk());
    step.max_leaf_residue_v = std::max(step.max_leaf_residue_v, push.get_FVk());
  }
  leaves.push_back({begin, end, start, finish});
  return true;
}

}  // namespace drift_kinetic
