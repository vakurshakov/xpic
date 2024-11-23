#ifndef SRC_BASIC_PARTICLES_PARTICLES_H
#define SRC_BASIC_PARTICLES_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/particles.h"
#include "src/utils/particle_shape.h"

namespace basic {

class Simulation;

class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const Sort_parameters& parameters);
  ~Particles() override;

  PetscErrorCode push();

private:
  static constexpr int OMP_CHUNK_SIZE = 16;

  void interpolate(const Shape& shape, Vector3R& E_p, Vector3R& B_p) const;
  void push(const Vector3R& E_p, const Vector3R& B_p, Point& point) const;
  void decompose(const Shape& shape, const Point& point);

  Simulation& simulation_;
  Vec local_E;
  Vec local_B;
  Vec local_J;
  Vector3R*** E;
  Vector3R*** B;
  Vector3R*** J;
};

}  // namespace basic

#endif  // SRC_BASIC_PARTICLES_PARTICLES_H
