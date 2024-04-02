#include "particles.h"

#include "src/impls/ricketson/simulation.h"

namespace ricketson {

Particles::Particles(const Particles_parameters& parameters) {
  PetscFunctionBeginUser;
  parameters_ = parameters;
  PetscFunctionReturnVoid();
}

PetscErrorCode Particles::add_particle(const Point& point) {
  PetscFunctionBeginUser;
  const Vector3R& r = point.r;
  points_.emplace_back(point);
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::push() {
  PetscFunctionBeginUser;

  for (auto it = points_.begin(); it != points_.end(); ++it) {
    Vector3R point_E = 0.0;
    Vector3R point_B = 0.0;

    PetscCall(push(point_E, point_B, *it));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::push(const Vector3R& point_E, const Vector3R& point_B, Point& point) const {
  PetscReal alpha = 0.5 * dt * charge(point);
  PetscReal m = mass(point);

  Vector3R& r = point.r;
  Vector3R& p = point.p;

  const Vector3R w = p + point_E * alpha;
  PetscReal energy = sqrt(m * m + w.dot(w));

  const Vector3R h = point_B * alpha / energy;
  const Vector3R s = h * 2.0 / (1.0 + h.dot(h));
  p = point_E * alpha + w * (1.0 - h.dot(s)) + w.cross(s) + h * (s.dot(w));

  energy = sqrt(m * m + p.dot(p));
  r += p * dt / energy;

  if (geom_nx == 1) r.x() = 0.5;
  if (geom_ny == 1) r.y() = 0.5;
  if (geom_nz == 1) r.z() = 0.5;
}

}
