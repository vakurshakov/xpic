#ifndef SRC_ECSIM_PARTICLES_H
#define SRC_ECSIM_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/particles.h"
#include "src/utils/shape.h"

namespace ecsim {

class Simulation;

class Particles : public interfaces::Particles {
public:
  DEFAULT_MOVABLE(Particles);

  Particles(Simulation& simulation, const SortParameters& parameters);
  ~Particles() override;

  PetscErrorCode calculate_energy();
  PetscErrorCode clear_sources();
  PetscErrorCode first_push();
  PetscErrorCode second_push();
  PetscErrorCode final_update();

  void fill_ecsim_current(PetscInt g, PetscReal* coo_v);

  Vector3R*** E;
  Vector3R*** B;

  Vec local_currI;
  Vec global_currI;
  Vector3R*** currI;

private:
  static constexpr const auto& shape_func = spline_of_1st_order;
  static constexpr const auto& shape_radius = 1.0;

  void decompose_ecsim_current(const Shape& shape, const Point& point,
    const Vector3R& B_p, PetscReal* coo_v);

  Simulation& simulation_;
};

}  // namespace ecsim

#endif  // SRC_ECSIM_PARTICLES_H
