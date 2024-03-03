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

  void interpolate(const Vector3<PetscReal>& r0, Vector3<PetscReal>& local_E, Vector3<PetscReal>& local_B) const;
  void push(Point& point, const Vector3<PetscReal>& local_E, const Vector3<PetscReal>& local_B) const;

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

  const Simulation& simulation_;

  const PetscMPIInt* neighbours;
  Vector3<PetscReal> l_start;
  Vector3<PetscReal> l_end;

  Particles_parameters parameters_;
  std::vector<Point> points_;
};

}

#endif  // SRC_BASIC_PARTICLES_PARTICLES_H
