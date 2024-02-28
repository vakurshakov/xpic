#ifndef SRC_INTERFACES_PARTICLES_PARTICLES_H
#define SRC_INTERFACES_PARTICLES_PARTICLES_H

#include "src/pch.h"

#include <petscdm.h>

#include "src/vectors/vector_classes.h"
#include "src/interfaces/particles/particle.h"
#include "src/interfaces/particles/parameters.h"


namespace basic {

class Simulation;

class Particles {
  static constexpr int MPI_TAG_NUMBERS = 2;
  static constexpr int MPI_TAG_PARTICLES = 4;
  static constexpr int OMP_CHUNK_SIZE = 16;
public:
  Particles(const Simulation& simulation, const Particles_parameters& parameters);

  PetscErrorCode add_particle(const Vector3<PetscReal>& r, const Vector3<PetscReal>& p);
  PetscErrorCode communicate();

  PetscErrorCode push();

private:
  void push(Particle& particle, const Vector3<PetscReal>& local_E, const Vector3<PetscReal>& local_B) const;

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
  std::vector<Particle> particles_;
};

}

#endif  // SRC_INTERFACES_PARTICLES_PARTICLES_H
