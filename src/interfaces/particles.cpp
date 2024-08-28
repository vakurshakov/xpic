#include "particles.h"

#include "src/impls/basic/simulation.h"

namespace interfaces {

const Sort_parameters& Particles::parameters() const {
  return parameters_;
}

PetscInt Particles::particles_number(const Point& point) const {
  return parameters_.Np;
}

PetscReal Particles::density(const Point& point) const {
  return parameters_.n;
}

PetscReal Particles::charge(const Point& /* point */) const {
  return parameters_.q;
}

PetscReal Particles::mass(const Point& /* point */) const {
  return parameters_.m;
}

Vector3R Particles::velocity(const Point& point) const {
  const Vector3R& p = point.p;
  PetscReal m = mass(point);
  return p / sqrt(m * m + p.square());
}

}
