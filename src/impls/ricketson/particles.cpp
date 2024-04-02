#include "particles.h"

#include "src/impls/ricketson/simulation.h"

namespace ricketson {

Particles::Particles(Simulation& simulation, const Particles_parameters& parameters)
  : simulation_(simulation) {
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

    PetscCall(adaptive_time_stepping());
    PetscCall(push(point_E, point_B, *it));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::adaptive_time_stepping() {
  Vector3R E_p = 0.0;
  Vector3R B_p = 0.0;
  Vector3R DB_p = 0.0;
  Vector3R v_E = 0.0;

  PetscReal Omega = parameters_.q * B_p.length() / parameters_.m;

  PetscReal delta_perp; /* = rho             * DB_p_perp.length() / B_p.length() */
  PetscReal delta_par;  /* = (v_par / Omega) * DB_p_par .length()  / B_p.length() */
  PetscReal delta_E;    /* = (v_E / Omega)   * DB_p_perp.length() / B_p.length() */
}


PetscErrorCode Particles::push(const Vector3R& point_E, const Vector3R& point_B, Point& point) const {
  PetscReal alpha = 0.5 * dt * charge(point) / mass(point);

  Vector3R& r = point.r;
  Vector3R& p = point.p;

  p += alpha * (point_E + p * point_B);
  r += p * dt;
}

}
