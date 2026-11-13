#ifndef SRC_DRIFT_KINETIC_PARTICLES_H
#define SRC_DRIFT_KINETIC_PARTICLES_H

#include "src/interfaces/particles.h"
#include "src/impls/drift_kinetic/adaptive_substepping.h"

namespace drift_kinetic {

class Simulation;

/// @brief `PointByField` access to the raw arrays of `Particles::storage`.
struct DkPointArrays {
  PetscReal* r;
  PetscReal* p;
  PetscReal* p_parallel;
  PetscReal* mu_p;

  /// @param suffix `"_prev"` gives access to the saved states.
  DkPointArrays(
    const ParticleStorage::Attributes& f, const std::string& suffix = "")
    : r(f["r" + suffix]),
      p(f["p" + suffix]),
      p_parallel(f["p_parallel" + suffix]),
      mu_p(f["mu_p" + suffix])
  {
  }

  PointByField load(PetscInt i) const
  {
    return {Vector3R(r + 3 * i), Vector3R(p + 3 * i), p_parallel[i], mu_p[i]};
  }

  void store(PetscInt i, const PointByField& point) const
  {
    std::copy_n(point.r.data, 3, r + 3 * i);
    std::copy_n(point.p.data, 3, p + 3 * i);
    p_parallel[i] = point.p_parallel;
    mu_p[i] = point.mu_p;
  }
};

/// @brief Advances drift-kinetic particles, `storage` holds `r`, `p` (the
/// momentum of the guiding center step), `p_parallel` and `mu_p`.
class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const SortParameters& parameters);
  PetscErrorCode initialize() override;
  PetscErrorCode finalize() override;

  PetscErrorCode form_iteration();

  PetscErrorCode ingest(
    const Point* points, PetscInt n, PetscReal* energy = nullptr) override;
  PetscErrorCode extract(const std::function<bool(const Point&)>& predicate,
    std::vector<Point>* extracted, PetscReal* energy = nullptr) override;
  PetscErrorCode prepare_storage();
  PetscErrorCode kinetic_sums_local(KineticSums& sums) const override;
  PetscReal get_average_iteration_number() const;
  PetscInt get_max_iteration_number() const;
  PetscInt get_push_retries() const;
  PetscInt get_push_leaf_failures() const;
  PetscReal get_max_push_leaf_residue_r() const;
  PetscReal get_max_push_leaf_residue_v() const;

  const PushStatistics& get_push_stats() const;
  void set_substepping(bool enabled, PetscInt depth);

  void set_fail_on_any_push_nonconvergence(bool value);
  bool get_fail_on_any_push_nonconvergence() const;

  void set_field_tolerances(PetscReal atol, PetscReal rtol);

  const Simulation& simulation() const;

  void set_coord_is_gc(bool v);

  Vec M;      // Sort magnetization.
  Vec M_loc;  // Local magnetization with ghost cells.
  Arr M_arr;  // Writable local magnetization array.

  Arr Bn0_arr;  // Previous magnetic field array.
  Arr Bn1_arr;  // Predicted magnetic field array.

protected:
  PushStatistics push_stats;
  bool substepping = false;
  PetscInt max_substep_depth = 4;
  bool coord_is_gc = false;
  bool fail_on_any_push_nonconvergence = false;  // Any SNES evaluation.

  PetscReal field_atol = 0;  // SNES tolerances of the field residual.
  PetscReal field_rtol = 0;

  Simulation& simulation_;
};

}  // namespace drift_kinetic

#endif  // SRC_DRIFT_KINETIC_PARTICLES_H
