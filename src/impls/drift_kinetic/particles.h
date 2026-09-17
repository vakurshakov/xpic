#ifndef SRC_DRIFT_KINETIC_PARTICLES_H
#define SRC_DRIFT_KINETIC_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/particles.h"

/// @file particles.h
/// @brief Drift-kinetic particle storage and advancement.

namespace drift_kinetic {

class Simulation;
class InjectParticles;
class SetPairedParticles;

/// @brief Converts a particle whose coordinate is already a guiding center.
PointByField make_point_at_gc(
  const Point& point, const Vector3R& Bp, PetscReal mp);

/// @brief Stores, advances and redistributes drift-kinetic particles.
class Particles : public interfaces::Particles {
public:
  /// @brief Creates an empty particle sort for a simulation.
  Particles(Simulation& simulation, const SortParameters& parameters);
  /// @brief Releases particle fields and base particle resources.
  PetscErrorCode finalize() override;

  /// @brief Converts newly loaded kinetic particles to guiding-center form.
  PetscErrorCode sync_dk_curr_storage();
  /// @brief Saves particle states before the nonlinear field iteration.
  PetscErrorCode prepare_storage();
  /// @brief Advances particles and deposits current and magnetization.
  PetscErrorCode form_iteration();
  /// @brief Returns the local kinetic energy of the sort.
  PetscReal kinetic_energy_local() const;
  /// @brief Returns the average pusher iteration count.
  PetscReal get_average_iteration_number() const;
  /// @brief Returns the largest pusher iteration count.
  PetscInt get_max_iteration_number() const;
  /// @brief Returns the number of adaptive retries.
  PetscInt get_push_retries() const { return push_retries_; }
  /// @brief Returns the number of failed terminal substeps.
  PetscInt get_push_leaf_failures() const { return push_leaf_failures_; }
  /// @brief Returns the largest terminal coordinate residual.
  PetscReal get_max_push_leaf_residue_r() const
  {
    return max_push_leaf_residue_r_;
  }
  /// @brief Returns the largest terminal velocity residual.
  PetscReal get_max_push_leaf_residue_v() const
  {
    return max_push_leaf_residue_v_;
  }

  /// @brief Controls whether terminal pusher failures abort the solve.
  void set_fail_on_terminal_nonconvergence(bool value)
  {
    fail_on_terminal_nonconvergence_ = value;
  }
  /// @brief Reports whether terminal pusher failures abort the solve.
  bool fail_on_terminal_nonconvergence() const
  {
    return fail_on_terminal_nonconvergence_;
  }
  /// @brief Returns the guiding-center storage grouped by local cell.
  const std::vector<std::list<PointByField>>& get_dk_curr_storage() const
  {
    return dk_curr_storage;
  }

  /// @brief Returns the owning simulation.
  const Simulation& simulation() const { return simulation_; }

  /// @brief Selects whether loaded coordinates are already guiding centers.
  void set_coord_is_gc(bool v) { coord_is_gc_ = v; }
  /// @brief Reports whether loaded coordinates are guiding centers.
  bool coord_is_gc() const { return coord_is_gc_; }

  Vec M;      ///< Sort magnetization.
  Vec M_loc;  ///< Local magnetization with ghost cells.
  Arr M_arr;  ///< Writable local magnetization array.

  Arr Bn_arr;   ///< Previous magnetic field array.
  Arr Bn1_arr;  ///< Predicted magnetic field array.

protected:
  /// @brief Moves particles between local cells.
  PetscErrorCode update_cells_seq();
  /// @brief Exchanges particles between MPI subdomains.
  PetscErrorCode update_cells_mpi();
  /// @brief Applies periodic boundary conditions to one guiding center.
  PetscErrorCode correct_coordinates(PointByField& point);
  std::vector<std::list<PointByField>> dk_curr_storage;  ///< Current states by cell.
  std::vector<std::vector<PointByField>> dk_prev_storage;  ///< Saved states by cell.
  PetscInt size = 0;       ///< Local particle count.
  PetscReal avgit = 0.0;   ///< Average pusher iteration count.
  PetscInt maxit = 0;      ///< Largest pusher iteration count.
  PetscInt push_retries_ = 0;
  PetscInt push_leaf_failures_ = 0;
  PetscReal max_push_leaf_residue_r_ = 0.0;
  PetscReal max_push_leaf_residue_v_ = 0.0;
  bool fail_on_terminal_nonconvergence_ = false;
  bool coord_is_gc_ = false;
  Simulation& simulation_;


  friend class InjectParticles;
  friend class SetPairedParticles;
};

}  // namespace drift_kinetic

#endif  // SRC_DRIFT_KINETIC_PARTICLES_H
