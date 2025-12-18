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

  Vec M;
  Vec M_loc;
  Arr M_arr;

protected:
  std::vector<std::list<PointByField>> dk_curr_storage;
  std::vector<std::vector<PointByField>> dk_prev_storage;
  Simulation& simulation_;
};

}  // namespace drift_kinetic

#endif  // SRC_DRIFT_KINETIC_PARTICLES_H
