#ifndef SRC_BASIC_PARTICLES_PARTICLES_H
#define SRC_BASIC_PARTICLES_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/particles.h"
#include "src/utils/shape.h"

namespace basic {

class Simulation;

class Particles final : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const SortParameters& parameters);
  PetscErrorCode finalize() override;

  PetscErrorCode push();

  Vector3R*** E;
  Vector3R*** B;

  Vec global_J;
  Vec local_J;
  Vector3R*** J;

  Simulation& simulation_;

  static constexpr PetscReal shr = 1.5;
  static constexpr auto&& sfunc = spline_of_2nd_order;
};

}  // namespace basic

#endif  // SRC_BASIC_PARTICLES_PARTICLES_H
