#ifndef SRC_RICKETSON_PARTICLES_PARTICLES_H
#define SRC_RICKETSON_PARTICLES_PARTICLES_H

#include "src/pch.h"

#include <petscdm.h>

#include "src/vectors/vector_classes.h"
#include "src/interfaces/particles/particles.h"

namespace ricketson {

class Simulation;

class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const Particles_parameters& parameters);

  PetscErrorCode add_particle(const Point& point);
  const std::vector<Point>& get_points() const { return points_; }

  PetscErrorCode push();

private:
  PetscErrorCode adaptive_time_stepping();
  PetscErrorCode push(const Vector3R& point_E, const Vector3R& point_B, Point& point) const;

  Simulation& simulation_;
  std::vector<Point> points_;
};

}

#endif  // SRC_RICKETSON_PARTICLES_PARTICLES_H
