#include "particles.h"

#include "src/impls/ricketson/simulation.h"
#include "src/impls/simple_interpolation.h"

namespace ricketson {

/// @note SNES solver tolerances
static constexpr PetscReal atol = 1e-10;
static constexpr PetscReal rtol = 1e-10;
static constexpr PetscReal stol = 1e-10;
static constexpr PetscInt maxit = 100;
static constexpr PetscInt maxf  = 300;

/**
 * @brief Internal constants controlling the restrictions of the proposed scheme.
 * @details
 * α ∈ (0, 1), controls how close to the time-step restrictions one is willing to get.
 * β ∈ (0, 1), controls the region |v_E / u - 1| < β, where conserving effective force becomes discontinuous.
 * ε ∈ (0, 1), controls the maximum permissible fractional change in μ within a time-step.
 * Γ > 0, measures the accuracy with which we wish to resolve spatial variations in the magnetic field.
 * t_res -- The smallest timescale in the problem that we wish to resolve.
 */
static constexpr PetscReal alpha = 0.9;
static constexpr PetscReal beta  = 0.2;
static constexpr PetscReal eps   = 0.15;
static constexpr PetscReal gamma = 0.1;
static constexpr PetscReal t_res = 0.1;


Particles::Particles(Simulation& simulation, const Particles_parameters& parameters)
    : simulation_(simulation) {
  PetscFunctionBeginUser;
  parameters_ = parameters;

  context_.width = min(Vector3I(Geom_n), Vector3I(shape_width));

  /// @todo It'd be more reusable to place particle mover into separate class

  // Nonlinear solver should be created for each process.
  PetscCallVoid(SNESCreate(PETSC_COMM_SELF, &snes_));
  PetscCallVoid(SNESSetType(snes_, SNESNRICHARDSON));
  PetscCallVoid(SNESSetFunction(snes_, nullptr, Particles::form_Picard_iteration, &context_));
  PetscCallVoid(SNESSetTolerances(snes_, atol, rtol, stol, maxit, maxf));

  PetscCallVoid(VecCreate(PETSC_COMM_SELF, &solution_));
  PetscCallVoid(VecSetType(solution_, VECSEQ));
  PetscCallVoid(VecSetSizes(solution_, PETSC_DECIDE, solution_size));

  LOG_INFO("Nonlinear solver for \"{}\" is set, tolerances:", parameters_.sort_name);
  LOG_INFO("  atol = {} - absolute convergence tolerance", atol);
  LOG_INFO("  rtol = {} - relative convergence tolerance", rtol);
  LOG_INFO("  stol = {} - convergence tolerance in terms of the norm of the change in the solution between steps", stol);
  LOG_INFO("  maxit = {} - maximum number of iterations", maxit);
  LOG_INFO("  maxf  = {} - maximum number of function evaluations", maxf);
  LOG_FLUSH();
  PetscFunctionReturnVoid();
}


Particles::~Particles() {
  PetscFunctionBeginUser;
  PetscCallVoid(SNESDestroy(&snes_));
  PetscCallVoid(VecDestroy(&solution_));
  PetscFunctionReturnVoid();
}


PetscErrorCode Particles::add_particle(const Point& point) {
  PetscFunctionBeginUser;
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

  PetscCall(DMDAVecGetArrayRead(da, local_E, &context_.E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &context_.B));
  PetscCall(DMDAVecGetArrayRead(da, local_B_grad, &context_.B_grad));

  for (auto it = points_.begin(); it != points_.end(); ++it) {
    PetscCall(push(*it));
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &context_.E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &context_.B));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B_grad, &context_.B_grad));

  PetscCall(DMRestoreLocalVector(da, &local_E));
  PetscCall(DMRestoreLocalVector(da, &local_B));
  PetscCall(DMRestoreLocalVector(da, &local_B_grad));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::adaptive_time_stepping(const Point& point) {
  PetscFunctionBeginUser;
  const Vector3R& x_n = point.r;
  const Vector3R& v_n = point.p;

  /// @todo Remove code duplication
  static Node node(x_n);
  static Shape shape[2];
  PetscCall(fill_shape(node.g, node.r, context_.width, false, shape[0]));
  PetscCall(fill_shape(node.g, node.r, context_.width, true, shape[1]));

  static Vector3R E_p;
  static Vector3R B_p;
  static Vector3R DB_p;
  Simple_interpolation interpolation(context_.width, shape[0], shape[1]);
  PetscCall(interpolation.process(node.g, {{E_p, context_.E}}, {{B_p, context_.B}, {DB_p, context_.B_grad}}));

  Vector3R v_p = v_n.parallel_to(B_p);
  Vector3R v_t = v_n.transverse_to(B_p);

  Vector3R DB_pp = DB_p.parallel_to(B_p);
  Vector3R DB_pt = DB_p.transverse_to(B_p);

  // (E) -- related to ExB particle drift
  Vector3R v_E = E_p.cross(B_p) / B_p.square();

  // It's assumed that the movement is dominated by ExB drift `v_E`, gyration `u` and parallel velocity `v_p`.
  PetscReal u = (v_t - v_E).length();
  PetscReal B_norm = B_p.length();
  PetscReal Omega = charge(point) * B_norm / mass(point);
  PetscReal rho = u / Omega;

  PetscReal delta_t = rho                  * (DB_pt.length() / B_norm);
  PetscReal delta_p = v_p.length() / Omega * (DB_pp.length() / B_norm);
  PetscReal delta_E = v_E.length() / Omega * (DB_pt.length() / B_norm);

  PetscReal Omega_dt = alpha *
    std::min(M_SQRT2 / sqrt(delta_E + delta_p),
      std::min(Omega * t_res, gamma / delta_t));

  // (eh) -- estimate of gyration velocity on half time-step
  PetscReal u_eh = (v_p - v_E).length() / sqrt(1.0 + 0.25 * Omega_dt * Omega_dt);

  /// @todo Probably, some diagnostic is need here to understand the cases
  if (v_E.length() > (1.0 + beta) * u_eh) {
    context_.dt = Omega_dt / Omega;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (v_E.length() > (1.0 - beta) * u) {
    context_.dt = std::min(t_res, gamma / Omega);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  Omega_dt = alpha *
    std::min(2 * sqrt(Omega * t_res / M_PI),
      2 * M_SQRT2 * std::min(1.0 / sqrt(delta_t), 1.0 / sqrt(delta_p)));

  context_.dt = Omega_dt / Omega;
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::push(Point& point) {
  PetscFunctionBeginUser;
  PetscCall(adaptive_time_stepping(point));
  context_.q = charge(point);
  context_.m = mass(point);

  // Initial guess should be explicitly set before `SNESSolve()`.
  PetscReal *arr;
  PetscCall(VecGetArrayWrite(solution_, &arr));
  arr[0] = context_.x_n[X] = point.x();
  arr[1] = context_.x_n[Y] = point.y();
  arr[2] = context_.x_n[Z] = point.z();
  arr[3] = context_.v_n[X] = point.px();
  arr[4] = context_.v_n[Y] = point.py();
  arr[5] = context_.v_n[Z] = point.pz();
  PetscCall(VecRestoreArrayWrite(solution_, &arr));

  /// @todo check \Delta mu / mu upon completion.
  PetscCall(SNESSolve(snes_, nullptr, solution_));

  // Updating point only in case of convergence
  SNESConvergedReason reason;
  PetscCall(SNESGetConvergedReason(snes_, &reason));

  if (reason >= 0) {
    PetscCall(VecGetArray(solution_, &arr));
    point.x() = arr[0];
    point.y() = arr[1];
    point.z() = arr[2];
    point.px() = arr[3];
    point.py() = arr[4];
    point.pz() = arr[5];
    PetscCall(VecRestoreArray(solution_, &arr));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Evaluates nonlinear function F(x).
 * @param[in]  snes     the SNES context.
 * @param[in]  vx       input vector of k-th iteration.
 * @param[in]  context  optional user-defined context.
 * @param[out] vf       function vector to be evaluated.
 *
 * @note `SNESNRICHARDSON` will iterate the following: x^{k+1} = x^{k} - lambda * F(x^{k}),
 * where lambda -- damping coefficient, lambda = +1.0 by default (no damping).
 */
PetscErrorCode Particles::form_Picard_iteration(SNES snes, Vec vx, Vec vf, void* __context) {
  PetscFunctionBeginUser;
  auto* context = (Particles::Context*)__context;
  const Vector3R& x_n = context->x_n;
  const Vector3R& v_n = context->v_n;

  PetscReal q = context->q;
  PetscReal m = context->m;
  PetscReal _alpha = 0.5 * context->dt * q / m;

  const PetscReal* x;
  PetscCall(VecGetArrayRead(vx, &x));
  Vector3R x_nn = {x[0], x[1], x[2]};
  Vector3R v_nn = {x[3], x[4], x[5]};
  PetscCall(VecRestoreArrayRead(vx, &x));

  /// @todo We should probably limit recalculation of the shape in case of close `x_half` iterations.
  Vector3R x_h = 0.5 * (x_nn + x_n);
  Vector3R v_h = 0.5 * (v_nn + v_n);

  static Node node(x_h);
  static Shape shape[2];
  PetscCall(fill_shape(node.g, node.r, context->width, false, shape[0]));
  PetscCall(fill_shape(node.g, node.r, context->width, true, shape[1]));

  static Vector3R E_p;
  static Vector3R B_p;
  static Vector3R DB_p;
  Simple_interpolation interpolation(context->width, shape[0], shape[1]);
  PetscCall(interpolation.process(node.g, {{E_p, context->E}}, {{B_p, context->B}, {DB_p, context->B_grad}}));

  PetscReal mu = -0.125 * m * ((v_nn - v_n).parallel_to(B_p)).square() / B_p.length();

  Vector3R DB_pp = DB_p.parallel_to(B_p);
  Vector3R DB_pt = DB_p.transverse_to(B_p);

  Vector3R G_p = -mu * DB_pp;
  Vector3R G_t;

  Vector3R v_ht = v_h.transverse_to(B_p);
  Vector3R v_E = E_p.cross(B_p) / B_p.square();
  PetscReal u = (v_ht - v_E).length();

  if (u > v_E.length()) {
    G_t = 2.0 * mu * DB_pt;
  }
  else {
    Vector3R v_Ed = v_E.normalized();
    Vector3R f_tEd = DB_pt.dot(v_Ed) * v_Ed;

    PetscReal v_hpn = v_h.parallel_to(B_p).length();
    PetscReal v_htn = v_ht.length();
    PetscReal zeta = 2.0 / POW2(u / v_E.length());

    G_t = zeta * v_hpn / v_htn * G_p * v_Ed - mu * (zeta * f_tEd + (DB_pt - f_tEd) / (1.0 - 1.0 / zeta));
  }

  B_p += (G_p + G_t).cross(v_ht) / (q * v_ht.square());

  Vector3R a = v_n + _alpha * E_p;

  // velocity on a new half-step, v^{n+1/2, k+1}
  v_h = (a + _alpha * a.cross(B_p) + POW2(_alpha) * a.dot(B_p) * B_p) / (1.0 + POW2(_alpha * B_p.length()));

  PetscReal* f;
  PetscCall(VecGetArrayWrite(vf, &f));
  f[0] = x_nn[X] - (x_n[X] + dt * v_h[X]);
  f[1] = x_nn[Y] - (x_n[Y] + dt * v_h[Y]);
  f[2] = x_nn[Z] - (x_n[Z] + dt * v_h[Z]);

  f[3] = v_nn[X] - (2.0 * v_h[X] - v_n[X]);
  f[4] = v_nn[Y] - (2.0 * v_h[Y] - v_n[Y]);
  f[5] = v_nn[Z] - (2.0 * v_h[Z] - v_n[Z]);
  PetscCall(VecRestoreArrayWrite(vf, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}
