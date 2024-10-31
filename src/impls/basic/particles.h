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
  static constexpr int OMP_CHUNK_SIZE  = 16;

  void interpolate(const Vector3I& p_g, Shape& no, Shape& sh, Vector3R& point_E, Vector3R& point_B) const;
  void push(const Vector3R& point_E, const Vector3R& point_B, Point& point) const;

  void decompose(const Vector3I& p_g, Shape& old_shape, Shape& new_shape, const Point& point);

  using Compute_j = std::function<PetscReal(PetscInt, PetscInt, PetscInt, PetscReal*)>;
  void decompose_dir(const Vector3I& p_g, const Compute_j& compute_j, Axis dir);

  Simulation& simulation_;
  Vec local_E, local_B, local_J;
  Vector3R ***E, ***B, ***J;
};

}

#endif  // SRC_BASIC_PARTICLES_PARTICLES_H
