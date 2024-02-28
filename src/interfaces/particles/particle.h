#ifndef SRC_INTERFACES_PARTICLES_PARTICLE_H
#define SRC_INTERFACES_PARTICLES_PARTICLE_H

#include "src/pch.h"
#include "src/vectors/vector_classes.h"
#include "src/interfaces/particles/parameters.h"


class Particle {
public:
  Vector3<PetscReal> r = 0.0;
  Vector3<PetscReal> p = 0.0;

  Particle() = default;
  Particle(const Vector3<PetscReal>& r, const Vector3<PetscReal>& p, const Particles_parameters& parameters);

  Particle(const Particle& particle);
  Particle(Particle&& particle);

  Particle& operator=(const Particle& particle);
  Particle& operator=(Particle&& particle);

  constexpr PetscReal& x() { return r.x(); }
  constexpr PetscReal& y() { return r.y(); }
  constexpr PetscReal& z() { return r.z(); }

  constexpr PetscReal x() const { return r.x(); }
  constexpr PetscReal y() const { return r.y(); }
  constexpr PetscReal z() const { return r.z(); }

  constexpr PetscReal& px() { return p.x(); }
  constexpr PetscReal& py() { return p.y(); }
  constexpr PetscReal& pz() { return p.z(); }

  constexpr PetscReal px() const { return p.x(); }
  constexpr PetscReal py() const { return p.y(); }
  constexpr PetscReal pz() const { return p.z(); }

  constexpr PetscReal n() const { return parameters->n; }
  constexpr PetscReal q() const { return parameters->q; }
  constexpr PetscReal m() const { return parameters->m; }

  Vector3<PetscReal> velocity() const {
    return p / sqrt(m() * m() + p.square());
  }

private:
  const Particles_parameters* parameters = nullptr;
};

void g_bound_reflective(Particle& particle, Axis axis);
void g_bound_periodic(Particle& particle, Axis axis);

#endif  // SRC_INTERFACES_PARTICLES_PARTICLE_H
