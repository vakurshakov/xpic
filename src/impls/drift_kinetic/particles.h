#ifndef SRC_DRIFT_KINETIC_PARTICLES_H
#define SRC_DRIFT_KINETIC_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/particles.h"
#include "src/utils/shape.h"

namespace drift_kinetic {

class Simulation;
class InjectParticles;
class SetPairedParticles;

/// @brief Builds a `PointByField` treating `point.r` as the guiding center
/// (no Larmor shift). Parallel momentum, perpendicular momentum and magnetic
/// moment are still computed from the locally interpolated `Bp`.
PointByField make_point_at_gc(
  const Point& point, const Vector3R& Bp, PetscReal mp);

class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const SortParameters& parameters);
  PetscErrorCode initialize_point_by_field(const Arr B_arr);
  PetscErrorCode finalize() override;

  PetscErrorCode sync_dk_curr_storage();
  PetscErrorCode prepare_storage();
  PetscErrorCode form_iteration();

  PetscReal kinetic_energy_local() const;
  PetscReal get_average_iteration_number() const;
  PetscInt get_max_iteration_number() const;
  const std::vector<std::list<PointByField>>& get_dk_curr_storage() const
  {
    return dk_curr_storage;
  }

  void set_coord_is_gc(bool v) { coord_is_gc_ = v; }
  bool coord_is_gc() const { return coord_is_gc_; }

  Vec M;
  Vec M_loc;
  Arr M_arr;

  Arr Bn_arr;
  Arr Bn1_arr;

protected:
  PetscReal n_Np(const PointByField& point) const;
  PetscReal qn_Np(const PointByField& point) const;
  PetscErrorCode update_cells_seq();
  PetscErrorCode update_cells_mpi();
  PetscErrorCode correct_coordinates(PointByField& point);
  std::vector<std::list<PointByField>> dk_curr_storage;
  std::vector<std::vector<PointByField>> dk_prev_storage;
  PetscInt size = 0;
  PetscReal avgit = 0.0;
  PetscInt maxit = 0;
  bool coord_is_gc_ = false;
  Simulation& simulation_;

  friend class InjectParticles;
  friend class SetPairedParticles;
};

}  // namespace drift_kinetic

#endif  // SRC_DRIFT_KINETIC_PARTICLES_H
