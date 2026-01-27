#ifndef SRC_DRIFT_KINETIC_PARTICLES_H
#define SRC_DRIFT_KINETIC_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/particles.h"
#include "src/utils/shape.h"

namespace drift_kinetic {

class Simulation;

class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const SortParameters& parameters);
  PetscErrorCode finalize() override;
  PetscErrorCode prepare_storage();
  PetscErrorCode form_iteration();
  PetscErrorCode initialize_point_by_field(const Arr B_arr);

  Vec M;
  Vec M_loc;
  Arr M_arr;

protected:
  PetscReal n_Np(const PointByField& point) const;
  PetscReal qn_Np(const PointByField& point) const;
  PetscErrorCode update_cells_seq();
  PetscErrorCode update_cells_mpi();
  PetscErrorCode correct_coordinates(PointByField& point);
  std::vector<std::list<PointByField>> dk_curr_storage;
  std::vector<std::vector<PointByField>> dk_prev_storage;
  Simulation& simulation_;
};

}  // namespace drift_kinetic

#endif  // SRC_DRIFT_KINETIC_PARTICLES_H
