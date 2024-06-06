#ifndef SRC_RICKETSON_PARTICLES_PARTICLES_H
#define SRC_RICKETSON_PARTICLES_PARTICLES_H

#include "src/pch.h"

#include <petscdm.h>

#include "src/vectors/vector3.h"
#include "src/interfaces/particles/particles.h"

#include "src/impls/particle_shape.h"

namespace ricketson {

class Simulation;

class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const Particles_parameters& parameters);

  PetscErrorCode add_particle(const Point& point);
  const std::vector<Point>& get_points() const { return points_; }

  PetscErrorCode push();

private:
  PetscErrorCode interpolate(const Vector3I& p_g, Shape& no, Shape& sh, Vector3R& point_E, Vector3R& point_B, Vector3R& point_DB) const;

  PetscErrorCode adaptive_time_stepping(const Vector3R& point_E, const Vector3R& point_B, const Vector3R& point_DB, const Point& point) const;
  PetscErrorCode push(const Vector3R& point_E, const Vector3R& point_B, Point& point) const;

  std::vector<Point> points_;

  Simulation& simulation_;
  Vec local_E, local_B, local_B_grad;
  Vector3R ***E, ***B, ***B_grad;

  Vector3I l_width;
};

}

#endif  // SRC_RICKETSON_PARTICLES_PARTICLES_H
