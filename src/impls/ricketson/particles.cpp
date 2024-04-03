#include "particles.h"

#include "src/impls/ricketson/simulation.h"

namespace ricketson {

Particles::Particles(Simulation& simulation, const Particles_parameters& parameters)
  : simulation_(simulation) {
  PetscFunctionBeginUser;
  parameters_ = parameters;

  PetscInt size[3];
  PetscCallVoid(DMDAGetCorners(simulation_.da_, REP3(nullptr), REP3_A(&size)));

  /// @todo Make converters from other types and sizes
  l_width[X] = std::min(shape_width, size[X]);
  l_width[Y] = std::min(shape_width, size[Y]);
  l_width[Z] = std::min(shape_width, size[Z]);
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

  const DM& da = simulation_.da_;
  PetscCall(DMGetLocalVector(da, &local_E));
  PetscCall(DMGetLocalVector(da, &local_B));
  PetscCall(DMGetLocalVector(da, &local_B_grad));

  PetscCall(DMGlobalToLocal(da, simulation_.E_, INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, simulation_.B_, INSERT_VALUES, local_B));
  PetscCall(DMGlobalToLocal(da, simulation_.B_grad_, INSERT_VALUES, local_B_grad));

  PetscCall(DMDAVecGetArrayRead(da, local_E, &E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));
  PetscCall(DMDAVecGetArrayRead(da, local_B_grad, &B_grad));

  for (auto it = points_.begin(); it != points_.end(); ++it) {
    Node node(it->r);
    Shape shape[2];

    PetscCall(fill_shape(node.g, node.r, shape[0], false));
    PetscCall(fill_shape(node.g, node.r, shape[1], true));

    Vector3R point_E = 0.0;
    Vector3R point_B = 0.0;
    Vector3R point_DB = 0.0;
    PetscCall(interpolate(node.g, shape[0], shape[1], point_E, point_B, point_DB));

    PetscCall(adaptive_time_stepping(point_E, point_B, point_DB, *it));

    PetscCall(push(point_E, point_B, *it));
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B_grad, &B_grad));

  PetscCall(DMRestoreLocalVector(da, &local_E));
  PetscCall(DMRestoreLocalVector(da, &local_B));
  PetscCall(DMRestoreLocalVector(da, &local_B_grad));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::fill_shape(const Vector3I& p_g, const Vector3R& p_r, Shape& shape, bool shift) {
  PetscFunctionBeginUser;
  PetscReal g_x, g_y, g_z;

  #pragma omp simd collapse(Vector3I::dim)
  for (PetscInt z = 0; z < l_width[Z]; ++z) {
  for (PetscInt y = 0; y < l_width[Y]; ++y) {
  for (PetscInt x = 0; x < l_width[X]; ++x) {
    PetscInt i = ((z * shape_width + y) * shape_width + x);
    g_x = p_g[X] + x;
    g_y = p_g[Y] + y;
    g_z = p_g[Z] + z;

    if (shift) {
      g_x += 0.5;
      g_y += 0.5;
      g_z += 0.5;
    }
    shape(i, X) = shape_function(p_r.x() - g_x, X);
    shape(i, Y) = shape_function(p_r.y() - g_y, Y);
    shape(i, Z) = shape_function(p_r.z() - g_z, Z);
  }}}
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::interpolate(const Vector3I& p_g, Shape& no, Shape& sh, Vector3R& point_E, Vector3R& point_B, Vector3R& point_DB) const {
  PetscFunctionBeginUser;
  PetscInt g_x, g_y, g_z;

  for (PetscInt z = 0; z < l_width[Z]; ++z) {
  for (PetscInt y = 0; y < l_width[Y]; ++y) {
  for (PetscInt x = 0; x < l_width[X]; ++x) {
    PetscInt i = ((z * shape_width + y) * shape_width + x);
    g_x = p_g[X] + x;
    g_y = p_g[Y] + y;
    g_z = p_g[Z] + z;

    point_E.x() += E[g_z][g_y][g_x].x() * no(i, Z) * no(i, Y) * sh(i, X);
    point_E.y() += E[g_z][g_y][g_x].y() * no(i, Z) * sh(i, Y) * no(i, X);
    point_E.z() += E[g_z][g_y][g_x].z() * sh(i, Z) * no(i, Y) * no(i, X);

    Vector3R B_shape = {
      sh(i, Z) * sh(i, Y) * no(i, X),
      sh(i, Z) * no(i, Y) * sh(i, X),
      no(i, Z) * sh(i, Y) * sh(i, X),
    };

    point_B.x() += B[g_z][g_y][g_x].x() * B_shape.x();
    point_B.y() += B[g_z][g_y][g_x].y() * B_shape.y();
    point_B.z() += B[g_z][g_y][g_x].z() * B_shape.z();

    point_DB.x() += B_grad[g_z][g_y][g_x].x() * B_shape.x();
    point_DB.y() += B_grad[g_z][g_y][g_x].y() * B_shape.y();
    point_DB.z() += B_grad[g_z][g_y][g_x].z() * B_shape.z();
  }}}
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::adaptive_time_stepping(const Vector3R& point_E, const Vector3R& point_B, const Vector3R& point_DB, const Point& point) const {
  PetscFunctionBeginUser;
  /// @todo SOLVE THE CONFLICT IN POINT CLASS, IT'S NOT A MOMENTUM HERE
  Vector3R v = point.p;

  Vector3R v_p = v.parallel_to(point_B);
  Vector3R v_t = v.transverse_to(point_B);

  Vector3R DB_p = point_DB.parallel_to(point_B);
  Vector3R DB_t = point_DB.transverse_to(point_B);

  // (E) -- related to ExB particle drift
  Vector3R v_E = point_E.cross(point_B) / point_B.square();

  // it's assumed that the movement is dominated by ExB drift `v_E`, gyration `u` and parallel velocity `v_p`
  Vector3R u = v_t - v_E;

  PetscReal B_norm = point_B.length();
  PetscReal Omega = parameters_.q * B_norm / parameters_.m;
  PetscReal rho = u.length() / Omega;

  /// @note We should avoid division by B_norm and use reciprocals (?)
  PetscReal delta_t = rho                  * (DB_t.length() / B_norm);
  PetscReal delta_p = v_p.length() / Omega * (DB_p.length() / B_norm);
  PetscReal delta_E = v_E.length() / Omega * (DB_t.length() / B_norm);

  PetscReal Omega_dt = simulation_.alpha *
    std::min(M_SQRT2 / sqrt(delta_E + delta_p),
      std::min(Omega * simulation_.t_res, simulation_.gamma / delta_t));

  // (eh) -- estimate of gyration velocity on half time-step
  PetscReal u_eh = (v_p - v_E).length() / sqrt(1.0 + 0.25 * Omega_dt * Omega_dt);

  /// @todo Probably, some diagnostic is need here to understand the cases
  if (v_E.length() > (1.0 + simulation_.beta) * u_eh) {
    dt = Omega_dt / Omega;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (v_E.length() > (1.0 - simulation_.beta) * u.length()) {
    dt = std::min(simulation_.t_res, simulation_.gamma / Omega);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  Omega_dt = simulation_.alpha *
    std::min(2 * sqrt(Omega * simulation_.t_res / M_PI),
      2 * M_SQRT2 * std::min(1.0 / sqrt(delta_t), 1.0 / sqrt(delta_p)));

  dt = Omega_dt / Omega;
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::push(const Vector3R& point_E, const Vector3R& point_B, Point& point) const {
  PetscFunctionBeginUser;
  PetscReal alpha = 0.5 * dt * charge(point) / mass(point);

  Vector3R& r = point.r;
  Vector3R& p = point.p;

  /// @todo "If, upon completion of the time-step, it is found that the fractional
  /// change in magnetic moment exceeds epsilon, shrink the step size by a factor
  /// of (alpha * epsilon * mu / D_mu) and recompute the step."
  p += alpha * (point_E + p * point_B);
  r += p * dt;

  PetscFunctionReturn(PETSC_SUCCESS);
}

}
