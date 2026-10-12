#ifndef SRC_DRIFT_KINETIC_PARTICLES_H
#define SRC_DRIFT_KINETIC_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/particles.h"

namespace drift_kinetic {

class Simulation;
class InjectParticles;
class SetPairedParticles;

PointByField make_point_at_gc(
  const Point& point, const Vector3R& Bp, PetscReal mp);

/// @brief Stores, advances and redistributes drift-kinetic particles.
class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const SortParameters& parameters);
  PetscErrorCode finalize() override;

  PetscErrorCode form_iteration();

  PetscErrorCode sync_dk_curr_storage();
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
  const std::vector<std::list<PointByField>>& get_dk_curr_storage() const;

  const Simulation& simulation() const;

  void set_coord_is_gc(bool v);
  bool get_coord_is_gc() const;

  Vec M;      // Sort magnetization.
  Vec M_loc;  // Local magnetization with ghost cells.
  Arr M_arr;  // Writable local magnetization array.

  Arr Bn_arr;   // Previous magnetic field array.
  Arr Bn1_arr;  // Predicted magnetic field array.

protected:
  PetscErrorCode update_cells_seq();
  PetscErrorCode update_cells_mpi();
  PetscErrorCode correct_coordinates(PointByField& point);

  std::vector<std::list<PointByField>> dk_curr_storage;  // Current states by cell.
  std::vector<std::vector<PointByField>> dk_prev_storage;  // Saved states by cell.

  PetscInt size = 0;      // Local particle count.
  PetscReal avgit = 0.0;  // Average pusher iteration count.
  PetscInt maxit = 0;     // Largest pusher iteration count.
  PetscInt push_retries = 0;
  PetscInt push_leaf_failures = 0;
  PetscReal max_push_leaf_residue_r = 0.0;
  PetscReal max_push_leaf_residue_v = 0.0;

  bool fail_on_terminal_nonconvergence = false;
  bool coord_is_gc = false;

  Simulation& simulation_;

  friend class InjectParticles;
  friend class SetPairedParticles;
};

}  // namespace drift_kinetic

#endif  // SRC_DRIFT_KINETIC_PARTICLES_H
