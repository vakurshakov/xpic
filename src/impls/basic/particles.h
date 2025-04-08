#ifndef SRC_BASIC_PARTICLES_PARTICLES_H
#define SRC_BASIC_PARTICLES_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/particles.h"
#include "src/utils/shape.h"

namespace basic {

class Simulation;

class Particles final : public interfaces::Particles {
public:
  DEFAULT_MOVABLE(Particles);

  Particles(Simulation& simulation, const SortParameters& parameters);
  PetscErrorCode finalize() override;

  PetscErrorCode push();

  Vector3R*** E;
  Vector3R*** B;

  Vec global_J;
  Vec local_J;
  Vector3R*** J;

private:
  void interpolate(const Shape& shape, Vector3R& E_p, Vector3R& B_p) const;
  void push(const Vector3R& E_p, const Vector3R& B_p, Point& point) const;
  void decompose(const Shape& shape, const Point& point);

  Simulation& simulation_;
};

}  // namespace basic

#endif  // SRC_BASIC_PARTICLES_PARTICLES_H
