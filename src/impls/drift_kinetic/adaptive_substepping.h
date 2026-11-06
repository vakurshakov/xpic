#ifndef SRC_IMPLS_DRIFT_KINETIC_ADAPTIVE_SUBSTEPPING_H
#define SRC_IMPLS_DRIFT_KINETIC_ADAPTIVE_SUBSTEPPING_H

#include "src/interfaces/point.h"

class DriftKineticPush;

namespace drift_kinetic {

class DriftKineticEsirkepov;

struct PushStatistics {
  PetscReal average_iterations() const;
  void merge(const PushStatistics& other);
  PetscErrorCode finish(const PushStatistics& local, MPI_Comm comm,
    const std::string& sort_name, bool fail_on_terminal_nonconvergence);

  PetscInt retries = 0;
  PetscInt leaf_failures = 0;
  PetscInt iteration_sum = 0;
  PetscInt max_iterations = 0;
  PetscInt root_failures = 0;
  PetscInt depth = 0;
  PetscReal max_leaf_residue_r = 0;
  PetscReal max_leaf_residue_v = 0;
  PetscInt particles = 0;
  PetscInt failed_particles = 0;
  PetscInt leaves = 0;
  PetscInt invalid_particles = 0;
};

/// @brief Coordinates substeps using the caller's pusher and field exchange.
class AdaptiveSubstepping {
public:
  AdaptiveSubstepping(PetscReal dt, PetscReal charge_weight,
    PetscReal particle_weight, PetscInt max_depth, PushStatistics& stats);

  void process(PointByField& current, const PointByField& previous,
    DriftKineticPush& push, DriftKineticEsirkepov& util);

private:
  bool advance(PetscReal begin, PetscReal end, PointByField& finish,
    const PointByField& start, PetscInt depth, DriftKineticPush& push,
    DriftKineticEsirkepov& util, PushStatistics& step);

  struct Leaf {
    PetscReal begin, end;
    PointByField start, finish;
  };

  PetscReal dt;
  PetscReal charge_weight, particle_weight;
  PetscInt max_depth;
  PushStatistics& stats;
  std::vector<Leaf> leaves;
};

}  // namespace drift_kinetic

#pragma omp declare reduction(                                                   \
    drift_kinetic_stats : drift_kinetic::PushStatistics : omp_out.merge(omp_in)) \
  initializer(omp_priv = drift_kinetic::PushStatistics{})

#endif  // SRC_IMPLS_DRIFT_KINETIC_ADAPTIVE_SUBSTEPPING_H
