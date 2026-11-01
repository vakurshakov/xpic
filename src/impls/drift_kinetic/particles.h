#ifndef SRC_DRIFT_KINETIC_PARTICLES_H
#define SRC_DRIFT_KINETIC_PARTICLES_H

#include "src/interfaces/particles.h"

namespace drift_kinetic {

class Simulation;

/// @brief `PointByField` access to the raw arrays of `Particles::storage`.
struct DkPointArrays {
  PetscReal* r;
  PetscReal* p;
  PetscReal* p_parallel;
  PetscReal* mu_p;

  /// @param suffix `"_prev"` gives access to the saved states.
  DkPointArrays(const ParticleStorage::Fields& f, const std::string& suffix = "")
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

  PetscErrorCode ingest(const Point* points, PetscInt n) override;
  PetscErrorCode prepare_storage();
  PetscReal kinetic_energy_local() const;
  PetscReal get_average_iteration_number() const;
  PetscInt get_max_iteration_number() const;
  PetscInt get_push_retries() const;
  PetscInt get_push_leaf_failures() const;
  PetscReal get_max_push_leaf_residue_r() const;
  PetscReal get_max_push_leaf_residue_v() const;

  void set_fail_on_terminal_nonconvergence(bool value);
  bool get_fail_on_terminal_nonconvergence() const;

  const Simulation& simulation() const;

  void set_coord_is_gc(bool v);

  Vec M;      // Sort magnetization.
  Vec M_loc;  // Local magnetization with ghost cells.
  Arr M_arr;  // Writable local magnetization array.

  Arr Bn0_arr;  // Previous magnetic field array.
  Arr Bn1_arr;  // Predicted magnetic field array.

protected:
  PetscInt size = 0;    // Local particle count.
  PetscReal avgit = 0;  // Average pusher iteration count.
  PetscInt maxit = 0;   // Largest pusher iteration count.
  PetscInt retries = 0;
  PetscInt leaf_failures = 0;
  PetscReal max_leaf_residue_r = 0;
  PetscReal max_leaf_residue_v = 0;

  bool coord_is_gc = false;
  bool fail_on_terminal_nonconvergence = false;

  Simulation& simulation_;
};

}  // namespace drift_kinetic

#endif  // SRC_DRIFT_KINETIC_PARTICLES_H
