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
  static constexpr int OMP_CHUNK_SIZE = 16;
public:
  Particles(const Simulation& simulation, const Particles_parameters& parameters);

  void add_particle(const Vector3<PetscReal>& r, const Vector3<PetscReal>& p);
  void push();

private:
  void push(Particle& particle, const Vector3<PetscReal>& local_E, const Vector3<PetscReal>& local_B) const;

  const Simulation& simulation_;
  Particles_parameters parameters_;
  std::vector<Particle> particles_;
};

}

#endif  // SRC_INTERFACES_PARTICLES_PARTICLES_H
