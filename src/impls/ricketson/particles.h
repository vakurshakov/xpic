#ifndef SRC_RICKETSON_PARTICLES_H
#define SRC_RICKETSON_PARTICLES_H

#include "src/interfaces/particles/particles.h"

#include <petscsnes.h>

#include "src/pch.h"
#include "src/vectors/vector3.h"
#include "src/impls/particle_shape.h"

namespace ricketson {

class Simulation;

class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const Particles_parameters& parameters);
  ~Particles();

  PetscErrorCode add_particle(const Point& point);
  const std::vector<Point>& get_points() const { return points_; }

  PetscErrorCode push();

  struct Context {
    // Field information in array form
    Vector3R ***E;
    Vector3R ***B;
    Vector3R ***B_grad;

    // Particle information
    Vector3I width;
    Vector3R x_n;
    Vector3R v_n;
    PetscReal alpha;
  };

private:
  PetscErrorCode interpolate(const Vector3I& p_g, Shape& no, Shape& sh, Vector3R& point_E, Vector3R& point_B, Vector3R& point_DB) const;
  PetscErrorCode adaptive_time_stepping(const Vector3R& point_E, const Vector3R& point_B, const Vector3R& point_DB, const Point& point);
  PetscErrorCode push(Point& point);

  std::vector<Point> points_;

  Simulation& simulation_;
  Vec local_E;
  Vec local_B;
  Vec local_B_grad;

  // Nonlinear solver environment
  const PetscInt solution_size = 6;  // (3 coordinates) + (3 velocities)
  SNES snes_;
  Vec solution_;
  Vec function_;
  Context context_;
};

}

#endif  // SRC_RICKETSON_PARTICLES_H