#include "particles.h"

#include "src/impls/ricketson/simulation.h"

namespace ricketson {

/**
 * @brief Evaluates nonlinear function F(x).
 * @param[in]  snes     the SNES context.
 * @param[in]  x        input vector of k-th iteration.
 * @param[in]  context  optional user-defined context.
 * @param[out] f        function vector to be evaluated.
 *
 * @todo We should try to pass the function as a mover class method.
 */
PetscErrorCode FormFunction(SNES snes, Vec vx, Vec vf, void* context) {
  PetscFunctionBeginUser;

  /// @todo Temporary, should be solved via context.
  PetscReal alpha = dt * (-1.0 / 1.0);  // dt * q / m
  Vector3R x_n = 0.0;
  Vector3R v_n = 0.0;

  /// @todo On-particle fields should be interpolated from grid.
  Vector3R E_p = {0.0, 1.0, 0.0};
  Vector3R B_p = {0.0, 0.0, 1.0};

  const PetscReal* x;
  PetscCall(VecGetArrayRead(vx, &x));

  PetscReal* f;
  PetscCall(VecGetArray(vf, &f));

  /// @todo To be changed on a proper Picard iteration step.
  Vector3R v_half = {x[3], x[4], x[5]};
  f[0] = x_n[X] + dt * v_half[X];
  f[1] = x_n[Y] + dt * v_half[Y];
  f[2] = x_n[Z] + dt * v_half[Z];

  f[3] = v_n[X] + alpha * (E_p[X] + v_half.cross(B_p)[X]);
  f[4] = v_n[Y] + alpha * (E_p[Y] + v_half.cross(B_p)[Y]);
  f[5] = v_n[Z] + alpha * (E_p[Z] + v_half.cross(B_p)[Z]);

  PetscCall(VecRestoreArrayRead(vx, &x));
  PetscCall(VecRestoreArray(vf, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Evaluates Jacobian matrix J(x).
 * @param[in]  snes     the SNES context.
 * @param[in]  x        input vector at k-th iteration.
 * @param[in]  context  optional user-defined context.
 * @param[out] jacobian Jacobian matrix
 * @param[out] B        optionally different preconditioning matrix
 */
PetscErrorCode FormJacobian(SNES snes, Vec x, Mat jacobian, Mat B, void* context) {
  PetscFunctionBeginUser;

  /// @todo Temporarily set identity matrix
  PetscInt  i[6] = {0, 1, 2, 3, 4, 5};
  PetscReal v[6] = {1, 1, 1, 1, 1, 1};
  PetscCall(MatSetValues(B, 6, i, 6, i, v, INSERT_VALUES));

  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  if (jacobian != B) {
    PetscCall(MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


Particles::Particles(Simulation& simulation, const Particles_parameters& parameters)
  : simulation_(simulation) {
  PetscFunctionBeginUser;
  parameters_ = parameters;

  PetscCallVoid(DMDAGetCorners(simulation_.da_, REP3(nullptr), REP3_A(&l_width)));
  l_width = min(l_width, Vector3I(shape_width));

  /// @todo It'd be more reusable to place particle mover into separate class

  // Nonlinear solver should be created for each process.
  PetscCallVoid(SNESCreate(PETSC_COMM_SELF, &snes_));
  PetscCallVoid(SNESSetType(snes_, SNESNRICHARDSON));

  PetscCallVoid(VecCreate(PETSC_COMM_SELF, &solution_));
  PetscCallVoid(VecSetSizes(solution_, solution_size, solution_size));
  PetscCallVoid(VecDuplicate(solution_, &residue_));

  PetscCallVoid(MatCreate(PETSC_COMM_SELF, &jacobian_));
  PetscCallVoid(MatSetSizes(jacobian_, solution_size, solution_size, solution_size, solution_size));
  PetscCallVoid(MatSetUp(jacobian_));

  PetscCallVoid(SNESSetFunction(snes_, residue_, FormFunction, nullptr));
  PetscCallVoid(SNESSetJacobian(snes_, jacobian_, jacobian_, FormJacobian, nullptr));

  PetscFunctionReturnVoid();
}


Particles::~Particles() {
  PetscFunctionBeginUser;
  PetscCallVoid(SNESDestroy(&snes_));
  PetscCallVoid(VecDestroy(&solution_));
  PetscCallVoid(VecDestroy(&residue_));
  PetscCallVoid(MatDestroy(&jacobian_));
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
    PetscCall(push(*it));
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B_grad, &B_grad));

  PetscCall(DMRestoreLocalVector(da, &local_E));
  PetscCall(DMRestoreLocalVector(da, &local_B));
  PetscCall(DMRestoreLocalVector(da, &local_B_grad));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::interpolate(const Vector3I& p_g, Shape& no, Shape& sh, Vector3R& point_E, Vector3R& point_B, Vector3R& point_DB) const {
  PetscFunctionBeginUser;
  PetscInt g_x, g_y, g_z;

  for (PetscInt z = 0; z < l_width[Z]; ++z) {
  for (PetscInt y = 0; y < l_width[Y]; ++y) {
  for (PetscInt x = 0; x < l_width[X]; ++x) {
    g_x = p_g[X] + x;
    g_y = p_g[Y] + y;
    g_z = p_g[Z] + z;

    PetscInt i = Shape::index(x, y, z);
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


PetscErrorCode Particles::push(Point& point) const {
  PetscFunctionBeginUser;

  /// @note Initial guess should be explicitly set _before_ `SNESSolve()`.
  PetscReal *arr;
  PetscCall(VecGetArrayWrite(solution_, &arr));
  arr[0] = point.x();
  arr[1] = point.y();
  arr[2] = point.z();
  arr[3] = point.px();
  arr[4] = point.py();
  arr[5] = point.pz();
  PetscCall(VecRestoreArrayWrite(solution_, &arr));

  /// @todo This is a part of a context
  // Node node(it->r);
  // Shape shape[2];

  /// @todo Working with the solver context
  // fill_shape(node.g, node.r, l_width, false, shape[0]);
  // fill_shape(node.g, node.r, l_width, true, shape[1]);
  // PetscCall(interpolate(node.g, shape[0], shape[1], point_E, point_B, point_DB));

  /// @todo This should be performed inside the solver
  // PetscCall(adaptive_time_stepping(point_E, point_B, point_DB, *it));

  PetscCall(SNESSolve(snes_, nullptr, solution_));

  PetscFunctionReturn(PETSC_SUCCESS);
}

}
