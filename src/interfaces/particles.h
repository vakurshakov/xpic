#ifndef SRC_INTERFACES_PARTICLES_H
#define SRC_INTERFACES_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/point.h"
#include "src/interfaces/sort_parameters.h"
#include "src/utils/particle_storage.h"
#include "src/utils/world.h"

/**
 * @brief `Point` access to the raw `"r"` and `"p"` arrays of `ParticleStorage`.
 * @warning It exists only while the kernels take `Point`; it shrinks as the sorts
 * move to raw fields and is kept only if `Point` stays as a local value type.
 */
struct PointArrays {
  PetscReal* r;
  PetscReal* p;

  PointArrays(const ParticleStorage::Attributes& f, const std::string& suffix = "")
    : r(f["r" + suffix]), p(f["p" + suffix])
  {
  }

  Point load(PetscInt i) const
  {
    return {Vector3R(r + 3 * i), Vector3R(p + 3 * i)};
  }

  void store(PetscInt i, const Point& point) const
  {
    std::copy_n(point.r.data, 3, r + 3 * i);
    std::copy_n(point.p.data, 3, p + 3 * i);
  }
};

namespace interfaces {

class Particles {
public:
  Particles(const World& world, const SortParameters& parameters);
  virtual ~Particles() = default;

  /**
   * @brief Creates the particle resources, the derived sort calls it first.
   * @warning The derived sort registers `"r"`, `"p"` and its own fields of
   * `storage` (if any) and calls `storage.commit_attributes()` afterwards.
   */
  virtual PetscErrorCode initialize();

  /// @brief Releases particle resources, derived sorts release their own first.
  virtual PetscErrorCode finalize();

  /// @brief Reference to outer `interfaces::Simulation::world`.
  const World& world;
  const DM da;
  const DM da_rho;
  const DM da_mflux;

  Arr E_arr;
  Arr B_arr;

  Vec J;
  Vec J_loc;
  Arr J_arr;

  Vec rho[2];
  Vec rho_loc[2];
  PetscReal*** rho_arr[2];

  Vec mflux;
  Vec mflux_loc;
  PetscReal**** mflux_arr;

  /// @brief Particles parameters, constant throughout the whole simulation run.
  const SortParameters parameters;

  /// @brief Kinetic points `r` and `p`, grouped by cell by `update_cells()`.
  ParticleStorage storage;

  /// @brief Whether the `point` lies within the local MPI-boundaries.
  bool is_local(const Point& point) const;

  /**
   * @brief Adds the kinetic `points` to `storage` and restores its cell order.
   * Drift-kinetic sorts convert them to their own representation.
   *
   * @param points Array of `n` points, all within the local MPI-boundaries
   * (see `is_local()`).
   * @note Collective, it is called by all ranks even with `n == 0`.
   */
  virtual PetscErrorCode ingest(const Point* points, PetscInt n);

  /// @brief Removes from `storage` the particles whose kinetic `Point` passes
  /// the `predicate` and appends them to `extracted`.
  PetscErrorCode extract(                                //
    const std::function<bool(const Point&)>& predicate,  //
    std::vector<Point>* extracted);

  /// @brief Hands over the particles that left the local box and sorts the
  /// rest by cell, `storage.cell_offsets()` are valid until the next push.
  virtual PetscErrorCode update_cells();

  PetscErrorCode calculate_moments();

  Vector3R velocity(const Point& point) const;

  PetscErrorCode log_distribution() const;

protected:
  static constexpr PetscInt MPI_TAG_NUMBERS = 2;
  static constexpr PetscInt MPI_TAG_POINTS = 4;
  static constexpr PetscInt OMP_CHUNK_SIZE = 16;

  PetscClassId classid;
  PetscLogEvent events[1];
};

}  // namespace interfaces

using Particles_sp = std::shared_ptr<interfaces::Particles>;

#endif  // SRC_INTERFACES_PARTICLES_H
