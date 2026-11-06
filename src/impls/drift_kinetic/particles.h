#ifndef SRC_DRIFT_KINETIC_PARTICLES_H
#define SRC_DRIFT_KINETIC_PARTICLES_H

#include "src/interfaces/particles.h"
#include "src/impls/drift_kinetic/adaptive_substepping.h"

namespace drift_kinetic {

class Simulation;

/// @brief Stores, advances and redistributes drift-kinetic particles.
class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const SortParameters& parameters);
  PetscErrorCode finalize() override;

  PetscErrorCode form_iteration();

  PetscErrorCode commit_added_particles() override;
  PetscErrorCode prepare_storage();
  PetscReal kinetic_energy_local() const;
  PetscReal get_average_iteration_number() const;
  PetscInt get_max_iteration_number() const;
  PetscInt get_push_retries() const;
  PetscInt get_push_leaf_failures() const;
  PetscReal get_max_push_leaf_residue_r() const;
  PetscReal get_max_push_leaf_residue_v() const;

  const PushStatistics& get_push_stats() const;
  void set_substepping(bool enabled, PetscInt depth);

  void set_fail_on_terminal_nonconvergence(bool value);
  bool get_fail_on_terminal_nonconvergence() const;
  const std::vector<std::list<PointByField>>& get_dk_curr_storage() const;

  const Simulation& simulation() const;

  void set_coord_is_gc(bool v);

  Vec M;      // Sort magnetization.
  Vec M_loc;  // Local magnetization with ghost cells.
  Arr M_arr;  // Writable local magnetization array.

  Arr Bn0_arr;  // Previous magnetic field array.
  Arr Bn1_arr;  // Predicted magnetic field array.

protected:
  PetscErrorCode update_cells_seq();
  PetscErrorCode update_cells_mpi();
  PetscErrorCode correct_coordinates(PointByField& point);

  std::vector<std::list<PointByField>> dk_curr_storage;  // Current states by cell.
  std::vector<std::vector<PointByField>> dk_prev_storage;  // Saved states by cell.

  PushStatistics push_stats;
  bool substepping = false;
  PetscInt max_substep_depth = 4;
  bool coord_is_gc = false;
  bool fail_on_terminal_nonconvergence = false;

  Simulation& simulation_;
};

}  // namespace drift_kinetic

#endif  // SRC_DRIFT_KINETIC_PARTICLES_H
