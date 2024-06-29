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


Particles::Particles(Simulation& simulation, const Particles_parameters& parameters) : simulation_(simulation) {
  PetscFunctionBeginUser;
  parameters_ = parameters;

  particle_iterations_log = Sync_binary_file(CONFIG().out_dir, "particle_iterations");

  ctx.width = min(Vector3I(Geom_n), Vector3I(shape_width));

  /// @todo It'd be more reusable to place particle mover into separate class

  // Nonlinear solver should be created for each process.
  PetscCallVoid(SNESCreate(PETSC_COMM_SELF, &snes_));
  PetscCallVoid(SNESSetType(snes_, SNESNRICHARDSON));
  PetscCallVoid(SNESSetFunction(snes_, nullptr, Particles::form_Picard_iteration, &ctx));
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


Particles::Particles(Particles&& other) : simulation_(other.simulation_) {
  points_ = std::move(other.points_);
  particle_iterations_log = std::move(other.particle_iterations_log);

  snes_ = other.snes_;
  solution_ = other.solution_;
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
  PetscCall(DMGetLocalVector(da, &local_DB));

  PetscCall(DMGlobalToLocal(da, simulation_.E_, INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, simulation_.B_, INSERT_VALUES, local_B));
  PetscCall(DMGlobalToLocal(da, simulation_.DB_, INSERT_VALUES, local_DB));

  PetscCall(DMDAVecGetArrayRead(da, local_E, &ctx.E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &ctx.B));
  PetscCall(DMDAVecGetArrayRead(da, local_DB, &ctx.DB));

  for (auto it = points_.begin(); it != points_.end(); ++it) {
    PetscCall(push(*it));
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &ctx.E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &ctx.B));
  PetscCall(DMDAVecRestoreArrayRead(da, local_DB, &ctx.DB));

  PetscCall(DMRestoreLocalVector(da, &local_E));
  PetscCall(DMRestoreLocalVector(da, &local_B));
  PetscCall(DMRestoreLocalVector(da, &local_DB));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::push(Point& point) {
  PetscFunctionBeginUser;
  ctx.x_n = point.r;
  ctx.v_n = point.p;
  ctx.m = mass(point);
  ctx.q = charge(point);
  PetscCall(ctx.update(point.r, point.p, true));
  PetscCall(adaptive_time_stepping(point));

  PetscReal mu_0 = (ctx.v_n - ctx.v_E).square() / ctx.B_p.length();

  const PetscInt MAX_ITERATIONS_RESTART = 10;
  for (PetscInt i = 0; i < MAX_ITERATIONS_RESTART; ++i) {
    // Initial guess should be explicitly set before `SNESSolve()`.
    PetscReal *arr;
    PetscCall(VecGetArrayWrite(solution_, &arr));
    arr[0] = ctx.x_n[X];
    arr[1] = ctx.x_n[Y];
    arr[2] = ctx.x_n[Z];
    arr[3] = ctx.v_n[X];
    arr[4] = ctx.v_n[Y];
    arr[5] = ctx.v_n[Z];
    PetscCall(VecRestoreArrayWrite(solution_, &arr));

    PetscCall(SNESSolve(snes_, nullptr, solution_));

    SNESConvergedReason reason;
    PetscCall(SNESGetConvergedReason(snes_, &reason));

    /// @todo Whether the context is evaluated at the last point x_nn?
    PetscReal mu = (point.p - ctx.v_E).square() / ctx.B_p.length();
    if (reason >= 0 && abs(mu - mu_0) / mu_0 < eps) {
      PetscCall(VecGetArray(solution_, &arr));
      point.x() = arr[0];
      point.y() = arr[1];
      point.z() = arr[2];
      point.px() = arr[3];
      point.py() = arr[4];
      point.pz() = arr[5];
      PetscCall(VecRestoreArray(solution_, &arr));

      PetscReal Omega_dt = (ctx.q * ctx.B_p.length() / ctx.m) * ctx.dt;

      const PetscInt size = 9;
      const PetscReal data[size] = {(PetscReal)i, Omega_dt, mu, REP3_A(point.r), REP3_A(point.p)};
      PetscCall(particle_iterations_log.write_floats(size, data));

      PetscFunctionReturn(PETSC_SUCCESS);
    }

    ctx.dt *= (alpha * eps * mu_0 / abs(mu - mu_0));
  }
  PetscReal Omega_dt = (ctx.q * ctx.B_p.length() / ctx.m) * ctx.dt;
  LOG_WARN("Particle iterations did not converged! Last timestep is (Omega * dt) = {:.5f}.", Omega_dt);
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::Context::update(const Vector3R& x_nn, const Vector3R& v_nn, bool force) {
  PetscFunctionBeginUser;
  if ((x_h - 0.5 * (x_nn + x_n)).length() > update_tolerance || force) {
    x_h = 0.5 * (x_nn + x_n);

    node = Node(x_h);
    PetscCall(fill_shape(node.g, node.r, width, false, shape[0]));
    PetscCall(fill_shape(node.g, node.r, width, true, shape[1]));

    Simple_interpolation interpolation(width, shape[0], shape[1]);
    PetscCall(interpolation.process(node.g, {{E_p, E}}, {{B_p, B}, {DB_p, DB}}));

    DB_pp = DB_p.parallel_to(B_p);
    DB_pt = DB_p.transverse_to(B_p);

    v_E = E_p.cross(B_p) / B_p.square();
    v_En = v_E.length();
  }

  v_h = 0.5 * (v_nn + v_n);
  v_hp = v_h.parallel_to(B_p);
  v_ht = v_h.transverse_to(B_p);
  u = (v_ht - v_E).length();
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::adaptive_time_stepping(const Point& point) {
  PetscFunctionBeginUser;
  PetscReal B_norm = ctx.B_p.length();
  PetscReal Omega  = ctx.q * B_norm / ctx.m;
  PetscReal rho    = ctx.u / Omega;

  PetscReal delta_t = rho                       * (ctx.DB_pt.length() / B_norm);
  PetscReal delta_p = ctx.v_hp.length() / Omega * (ctx.DB_pp.length() / B_norm);
  PetscReal delta_E = ctx.v_En          / Omega * (ctx.DB_pt.length() / B_norm);

  PetscReal Omega_dt = alpha *
    std::min(M_SQRT2 / sqrt(delta_E + delta_p),
      std::min(Omega * t_res, gamma / delta_t));

  // (eh) -- estimate of gyration velocity on half time-step.
  PetscReal u_eh = (ctx.v_ht - ctx.v_E).length() / sqrt(1.0 + 0.25 * Omega_dt * Omega_dt);

  /// @todo Probably, some diagnostic is need here to understand the cases.
  if (ctx.v_En > (1.0 + beta) * u_eh) {
    ctx.dt = Omega_dt / Omega;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (ctx.v_En > (1.0 - beta) * ctx.u) {
    ctx.dt = std::min(t_res, gamma / Omega);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  Omega_dt = alpha *
    std::min(2.0 * sqrt(Omega * t_res / M_PI),
      2.0 * M_SQRT2 * std::min(1.0 / sqrt(delta_t), 1.0 / sqrt(delta_p)));

  ctx.dt = Omega_dt / Omega;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Evaluates nonlinear function F(x).
 * @param[in]  snes the SNES context.
 * @param[in]  vx   input vector of k-th iteration.
 * @param[in]  ctx  optional user-defined context.
 * @param[out] vf   function vector to be evaluated.
 *
 * @note `SNESNRICHARDSON` will iterate the following: x^{k+1} = x^{k} - lambda * F(x^{k}),
 * where lambda -- damping coefficient, lambda = +1.0 by default (no damping).
 */
PetscErrorCode Particles::form_Picard_iteration(SNES snes, Vec vx, Vec vf, void* __ctx) {
  PetscFunctionBeginUser;
  auto& ctx = *(Particles::Context*)__ctx;
  const Vector3R& x_n = ctx.x_n;
  const Vector3R& v_n = ctx.v_n;

  const PetscReal* x;
  PetscCall(VecGetArrayRead(vx, &x));
  Vector3R x_nn = {x[0], x[1], x[2]};
  Vector3R v_nn = {x[3], x[4], x[5]};
  PetscCall(VecRestoreArrayRead(vx, &x));

  PetscCall(ctx.update(x_nn, v_nn));

  // To avoid the update of `Context::B_p`.
  Vector3R B_p = ctx.B_p;

  /// @todo Add guards in case of v_t << v_p
  PetscReal mu = -0.125 * ctx.m * ((v_nn - v_n).parallel_to(B_p)).square() / B_p.length();
  Vector3R G_p = -mu * ctx.DB_pp;
  Vector3R G_t;

  if (ctx.u > ctx.v_En) {
    G_t = 2.0 * mu * ctx.DB_pt;
  }
  else {
    PetscReal zeta = 2.0 / POW2(ctx.u / ctx.v_En);
    PetscReal v_hpn = ctx.v_hp.length();
    PetscReal v_htn = ctx.v_ht.length();

    Vector3R v_Ed = ctx.v_E.normalized();
    Vector3R f_pE = ctx.DB_pt.parallel_to(v_Ed);
    Vector3R f_tE = ctx.DB_pt.transverse_to(v_Ed);

    G_t = zeta * v_hpn / v_htn * G_p * v_Ed - mu * (zeta * f_pE + f_tE / (1.0 - 1.0 / zeta));
  }

  B_p += (G_p + G_t).cross(ctx.v_ht) / (ctx.q * ctx.v_ht.square());

  PetscReal tqm = 0.5 * ctx.dt * ctx.q / ctx.m;
  Vector3R a = v_n + tqm * ctx.E_p;

  // Velocity on a new half-step, v^{n+1/2, k+1}.
  Vector3R v_h = (a + tqm * a.cross(B_p) + POW2(tqm) * a.dot(B_p) * B_p) / (1.0 + POW2(tqm * B_p.length()));

  PetscReal* f;
  PetscCall(VecGetArrayWrite(vf, &f));
  f[0] = x_nn[X] - (x_n[X] + ctx.dt * v_h[X]);
  f[1] = x_nn[Y] - (x_n[Y] + ctx.dt * v_h[Y]);
  f[2] = x_nn[Z] - (x_n[Z] + ctx.dt * v_h[Z]);

  f[3] = v_nn[X] - (2.0 * v_h[X] - v_n[X]);
  f[4] = v_nn[Y] - (2.0 * v_h[Y] - v_n[Y]);
  f[5] = v_nn[Z] - (2.0 * v_h[Z] - v_n[Z]);
  PetscCall(VecRestoreArrayWrite(vf, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}
