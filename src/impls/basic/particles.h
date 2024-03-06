#ifndef SRC_BASIC_PARTICLES_PARTICLES_H
#define SRC_BASIC_PARTICLES_PARTICLES_H

#include "src/pch.h"

#include <petscdm.h>

#include "src/vectors/vector_classes.h"
#include "src/interfaces/particles/particles.h"

namespace basic {

class Simulation;

class Particles : public interfaces::Particles {
public:
  Particles(const Simulation& simulation, const Particles_parameters& parameters);

  PetscErrorCode add_particle(const Point& point);

  PetscErrorCode push();
  PetscErrorCode communicate();

private:
  static constexpr int OMP_CHUNK_SIZE  = 16;

  struct Shape;
  void fill_shape(const Vector3<PetscReal>& r0, Shape& shape);

  void interpolate(const Vector3<PetscReal>& r0, Shape& shape, Vector3<PetscReal>& point_E, Vector3<PetscReal>& point_B) const;
  void push(const Vector3<PetscReal>& point_E, const Vector3<PetscReal>& point_B, Point& point) const;

  PetscInt to_contiguous_index(PetscInt x, PetscInt y, PetscInt z) {
    constexpr PetscInt dim = 3;
    return (z * dim + y) * dim + x;
  }

  void from_contiguous_index(PetscInt index, PetscInt& x, PetscInt& y, PetscInt& z) {
    constexpr PetscInt dim = 3;
    x = (index) % dim;
    y = (index / dim) % dim;
    z = (index / dim) / dim;
  }

  Particles_parameters parameters_;
  std::vector<Point> points_;

  const Simulation& simulation_;
  Vec local_E, local_B;
  Vector3<PetscReal> ***E, ***B;

  const PetscMPIInt* neighbours;
  Vector3<PetscReal> l_start;
  Vector3<PetscReal> l_end;
  Vector3<PetscInt>  l_width;
};

}

#endif  // SRC_BASIC_PARTICLES_PARTICLES_H
