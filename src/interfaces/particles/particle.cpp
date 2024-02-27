#include "particle.h"

Particle::Particle(
  const Vector3<PetscReal>& r,
  const Vector3<PetscReal>& p,
  const Particles_parameters& parameters)
  : r(r),
    p(p),
    parameters(&parameters) {}

Particle::Particle(const Particle& particle)
  : r(particle.r),
    p(particle.p),
    parameters(particle.parameters) {}

Particle::Particle(Particle&& particle)
  : r(std::move(particle.r)),
    p(std::move(particle.p)),
    parameters(particle.parameters) {
  particle.parameters = nullptr;
}

Particle& Particle::operator=(const Particle& particle) {
  r = particle.r;
  p = particle.p;
  parameters = particle.parameters;
  return *this;
}

Particle& Particle::operator=(Particle&& particle) {
  r = std::move(particle.r);
  p = std::move(particle.p);
  parameters = particle.parameters;
  particle.parameters = nullptr;
  return *this;
}


void g_bound_reflective(Particle& particle, Axis axis) {
  PetscReal& s = particle.r[axis];
  PetscReal& ps = particle.p[axis];

  if (s < 0.0) {
    s = 0.0;
    ps *= -1.0;
  }
  else if (s > Geom[axis]) {
    s = Geom[axis];
    ps *= -1.0;
  }
}

void g_bound_periodic(Particle& particle, Axis axis) {
  PetscReal& s = particle.r[axis];

  if (s < 0.0) {
    s = Geom[axis] - (0.0 - s);
  }
  else if (s > Geom[axis]) {
    s = 0.0 + (s - Geom[axis]);
  }
}
