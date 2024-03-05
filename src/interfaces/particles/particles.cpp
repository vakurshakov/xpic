#include "particles.h"

#include "src/impls/basic/simulation.h"

namespace interfaces {

const Particles_parameters& Particles::parameters() const {
  return parameters_;
}

PetscInt Particles::particles_number(const Point& point) const {
#if !PARTICLES_LOCAL_PNUM
  return parameters_.Np;
#else
  return point.__Np;
#endif
}

PetscReal Particles::density(const Point& point) const {
#if !PARTICLES_LOCAL_DENSITY
  return parameters_.n;
#else
  return point.__n;
#endif
}

PetscReal Particles::charge(const Point& /* point */) const {
  return parameters_.q;
}

PetscReal Particles::mass(const Point& /* point */) const {
  return parameters_.m;
}

Vector3<PetscReal> Particles::velocity(const Point& point) const {
  const Vector3<PetscReal>& p = point.p;
  PetscReal m = mass(point);
  return p / sqrt(m * m + p.square());
}

}
