#ifndef SRC_ECCAPFIM_PARTICLES_H
#define SRC_ECCAPFIM_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/particles.h"
#include "src/utils/shape.h"

namespace eccapfim {

class Simulation;

class Particles : public interfaces::Particles {
public:
  DEFAULT_MOVABLE(Particles);

  Particles(Simulation& simulation, const SortParameters& parameters);
  PetscErrorCode finalize() override;

  PetscErrorCode clear_sources();
  PetscErrorCode prepare_storage();
  PetscErrorCode form_iteration();

  Vector3R*** E;
  Vector3R*** B;

  Vec local_J;
  Vec global_J;
  Vector3R*** J;

protected:
  static constexpr const auto& shape_func = spline_of_2nd_order;
  static constexpr const auto& shape_radius = 1.5;

  /// @note We should iterate the `Point` ~ (x^{n+1,k}, v^{n+1,k}) from _previous_
  /// timestep, meaning that we have to store copy of `Particles::storage`.
  std::vector<std::vector<Point>> previous_storage;

  Simulation& simulation_;
};

}  // namespace eccapfim

#endif  // SRC_ECCAPFIM_PARTICLES_H
