#include "src/algorithms/drift_kinetic_push.h"
#include "src/algorithms/implicit_drift_kinetic.h"
#include "src/impls/drift_kinetic/adaptive_substepping.h"

namespace {

using namespace drift_kinetic;

constexpr PetscInt n = 16;
constexpr PetscReal tolerance = 2e-12;
constexpr PetscReal center = 7;

struct Grid {
  Vector3R values[n][n][n]{};
  Vector3R* rows[n][n];
  Vector3R** planes[n];

  Grid()
  {
    for (PetscInt z = 0; z < n; z++) {
      planes[z] = rows[z];
      for (PetscInt y = 0; y < n; y++)
        rows[z][y] = values[z][y];
    }
  }
};

#if DRIFT_KINETIC_AITKEN == 0
PetscReal shape(PetscReal x)
{
  x = std::abs(x);
  if (x <= 0.5) return 0.75 - x * x;
  if (x < 1.5) return 0.5 * (1.5 - x) * (1.5 - x);
  return 0;
}

PetscReal density(const PointByField& point, PetscInt x, PetscInt y, PetscInt z)
{
  return shape(x - point.r[X]) * shape(y - point.r[Y]) *
    shape(z - point.r[Z]);
}
#endif

PetscErrorCode run_case(bool oscillator, PetscInt depth)
{
  PetscFunctionBeginUser;
  Grid electric, magnetic, current, magnetization, expected_j, expected_m;
  for (PetscInt z = 0; z < n; z++)
    for (PetscInt y = 0; y < n; y++)
      for (PetscInt x = 0; x < n; x++) {
        electric.values[z][y][x][Z] = oscillator ? -4 * (z + 0.5 - center) : 0.2;
        magnetic.values[z][y][x][Z] = 1;
      }

  DriftKineticEsirkepov util(electric.planes, magnetic.planes, magnetic.planes,
    magnetic.planes, current.planes, magnetization.planes);
  DriftKineticPush push(1, 1);
  push.set_fields_callback([&](const Vector3R& start, const Vector3R& finish,
                             Vector3R& E, PetscReal& B, Vector3R& b,
                             Vector3R& grad, Vector3R& rot) {
    return util.interpolate(E, B, b, grad, rot, finish, start);
  });

  PointByField initial{};
  initial.r = {7.1, 7.2, 7.1};
  initial.p_parallel = 0.1;
  initial.mu_p = 0.3;
  PointByField result = initial;
  PushStatistics stats;
  AdaptiveSubstepping substeps(1, 1, 1, depth, stats);
  substeps.process(result, initial, push, util);

  PetscCheck(stats.invalid_particles == 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
    "Invalid particle in substep test");

  if (!oscillator) {
    PetscCheck(stats.retries == 0 && stats.leaves == 1 && stats.leaf_failures == 0,
      PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected split in constant field");
    PetscCheck(std::abs(result.r[Z] - (initial.r[Z] + 0.2)) < tolerance &&
        std::abs(result.p_parallel - 0.3) < tolerance,
      PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Constant-field trajectory is incorrect");

    PointByField direct = initial;
    util.set_interval(0, 1);
    PetscCall(push.process(1, direct, initial));
    PetscCheck(push.has_converged() && (direct.r - result.r).abs_max() < tolerance &&
        std::abs(direct.p_parallel - result.p_parallel) < tolerance,
      PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unsplit result differs from direct push");
    PetscFunctionReturn(PETSC_SUCCESS);
  }

#if DRIFT_KINETIC_AITKEN == 0
  if (depth == 0) {
    PetscCheck(stats.root_failures == 1 && stats.failed_particles == 1 &&
        stats.leaf_failures == 1 && stats.leaves == 1 && stats.retries == 0,
      PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Depth limit did not record non-convergence");
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCheck(stats.root_failures == 1 && stats.retries == 1 && stats.depth == 1 &&
      stats.leaves == 2 && stats.failed_particles == 0 && stats.leaf_failures == 0,
    PETSC_COMM_WORLD, PETSC_ERR_PLIB,
    "Expected one split and two converged leaves: root=%d splits=%d leaves=%d failures=%d depth=%d",
    (int)stats.root_failures, (int)stats.retries, (int)stats.leaves,
    (int)stats.leaf_failures, (int)stats.depth);

  DriftKineticEsirkepov reference(electric.planes, magnetic.planes,
    magnetic.planes, magnetic.planes, expected_j.planes, expected_m.planes);
  PointByField previous = initial;
  PetscReal position = initial.r[Z] - center;
  PetscReal velocity = initial.p_parallel;
  for (PetscInt i = 0; i < 2; i++) {
    const PetscReal next_position = (0.75 * position + 0.5 * velocity) / 1.25;
    const PetscReal next_velocity = (-2 * position + 0.75 * velocity) / 1.25;
    PointByField next = previous;
    next.r[Z] = center + next_position;
    next.p_parallel = next_velocity;
    reference.set_interval(0.5 * i, 0.5 * (i + 1));
    PetscCall(reference.decomposition(next.r, previous.r,
      (next.r - previous.r) / 0.5, 0.5, 0.5 * initial.mu_p));
    previous = next;
    position = next_position;
    velocity = next_velocity;
  }

  PetscCheck((result.r - previous.r).abs_max() < tolerance &&
      std::abs(result.p_parallel - velocity) < tolerance,
    PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Split trajectory differs from midpoint solution");

  PetscReal source_error = 0, continuity_error = 0;
  for (PetscInt z = 1; z < n; z++)
    for (PetscInt y = 1; y < n; y++)
      for (PetscInt x = 1; x < n; x++) {
        source_error = std::max(source_error,
          (current.values[z][y][x] - expected_j.values[z][y][x]).abs_max());
        source_error = std::max(source_error,
          (magnetization.values[z][y][x] - expected_m.values[z][y][x]).abs_max());
        const PetscReal div_j = current.values[z][y][x][X] - current.values[z][y][x - 1][X] +
          current.values[z][y][x][Y] - current.values[z][y - 1][x][Y] +
          current.values[z][y][x][Z] - current.values[z - 1][y][x][Z];
        continuity_error = std::max(continuity_error,
          std::abs(density(result, x, y, z) - density(initial, x, y, z) + div_j));
      }

  PetscCheck(source_error < tolerance && continuity_error < tolerance,
    PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Source error %g, continuity error %g",
    (double)source_error, (double)continuity_error);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
  dx = dy = dz = 1;
  for (Axis a : {X, Y, Z}) {
    Dx[a] = 1;
    Geom_n[a] = n;
  }
  PetscCall(run_case(false, 0));
  PetscCall(run_case(false, 4));
#if DRIFT_KINETIC_AITKEN == 0
  PetscCall(run_case(true, 0));
  PetscCall(run_case(true, 4));
#endif
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
