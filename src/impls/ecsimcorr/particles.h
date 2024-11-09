#ifndef SRC_ECSIMCORR_PARTICLES_H
#define SRC_ECSIMCORR_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/particles.h"
#include "src/utils/particle_shape.h"

namespace ecsimcorr {

class Simulation;

class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const Sort_parameters& parameters);
  ~Particles() override;

  PetscErrorCode reset();
  PetscErrorCode first_push();

private:
  static constexpr int OMP_CHUNK_SIZE = 16;

  static constexpr auto& shape_func1 = __1st_order_spline;
  static constexpr auto& shape_func2 = __2nd_order_spline;
  static constexpr PetscInt shape_width1 = 2;
  static constexpr PetscInt shape_width2 = 4;

  void decompose_esirkepov_current(const Vector3I& p_g, const Shape& old_shape,
    const Shape& new_shape, const Point& point);

  void first_interpolate(const Vector3I& p_g, const Shape& no, const Shape& sh,
    Vector3R& point_B) const;

  void decompose_identity_current(const Vector3I& p_g, const Shape& no,
    const Shape& sh, const Point& point, const Vector3R& point_B);

  Simulation& simulation_;
  Vec local_E;
  Vec local_B;
  Vec local_currI;
  Vec local_currJe;
  Vector3R*** E;
  Vector3R*** B;
  Vector3R*** J;
  Vector3R*** currI;
  Vector3R*** currJe;
};

}  // namespace ecsimcorr

#endif  // SRC_ECSIMCORR_PARTICLES_H
