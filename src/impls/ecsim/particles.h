#ifndef SRC_ECSIM_PARTICLES_H
#define SRC_ECSIM_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/particles.h"
#include "src/utils/shape.h"

namespace ecsim {

class Simulation;

class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const SortParameters& parameters);
  PetscErrorCode finalize() override;

  virtual PetscErrorCode clear_sources();
  virtual PetscErrorCode first_push();
  virtual PetscErrorCode second_push();

  PetscErrorCode fill_ecsim_current(PetscReal* coo_v);

  Vec currI;
  Vec currI_loc;
  Arr currI_arr;

protected:
  static constexpr const auto& shape_func1 = spline_of_1st_order;
  static constexpr const auto& shape_radius1 = 1.0;

  void decompose_ecsim_current(const Point& point, PetscReal* coo_v);

  Simulation& simulation_;
};

}  // namespace ecsim

#endif  // SRC_ECSIM_PARTICLES_H
