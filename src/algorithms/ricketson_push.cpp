#include "ricketson_push.h"

#if 0

  #include "src/algorithms/simple_interpolation.h"

void RicketsonPush::push(Point& point)
{
  PetscReal mu_0 = (v_n - v_E).squared() / B_p.length();

  const PetscInt MAX_ITERATIONS_RESTART = 10;
  for (PetscInt i = 0; i < MAX_ITERATIONS_RESTART; ++i) {
    // Initial guess should be explicitly set before `SNESSolve()`.
    PetscReal* arr;
    VecGetArrayWrite(solution_, &arr);
    arr[0] = x_n[X];
    arr[1] = x_n[Y];
    arr[2] = x_n[Z];
    arr[3] = v_n[X];
    arr[4] = v_n[Y];
    arr[5] = v_n[Z];
    VecRestoreArrayWrite(solution_, &arr);

    SNESSolve(snes_, nullptr, solution_);

    SNESConvergedReason reason;
    SNESGetConvergedReason(snes_, &reason);

    VecGetArray(solution_, &arr);
    point.x() = arr[0];
    point.y() = arr[1];
    point.z() = arr[2];
    point.px() = arr[3];
    point.py() = arr[4];
    point.pz() = arr[5];
    VecRestoreArray(solution_, &arr);

    PetscInt iterations;
    PetscInt evals;
    SNESGetIterationNumber(snes_, &iterations);
    SNESGetNumberFunctionEvals(snes_, &evals);
    LOG("Iterations number: {}, Function evaluations: {}", iterations, evals);

    update(point.r, point.p);
    PetscReal mu = (point.p - v_E).squared() / B_p.length();
    LOG("|dmu| = {}, eps*mu0 = {}, shrink = {}", abs(mu - mu_0), eps * mu_0 , alpha * eps * mu_0 / abs(mu - mu_0));

    if (reason >= 0 && abs(mu - mu_0) < eps * mu_0) {
      PetscFunctionReturn(PETSC_SUCCESS);
    }

    dt *= (alpha * eps * mu_0 / abs(mu - mu_0));
  }

  // Resetting to their inital values in case of no convergence.
  point.r = x_n;
  point.p = v_n;
}


void RicketsonPush::Context::update(
  const Vector3R& x_nn, const Vector3R& v_nn)
{
  PetscFunctionBeginUser;
  x_h = 0.5 * (x_nn + x_n);

  shape.setup(x_h);

  SimpleInterpolation interpolation(shape);
  interpolation.process({{E_p, E}}, {{B_p, B}, {DB_p, DB}});

  DB_pp = DB_p.parallel_to(B_p);
  DB_pt = DB_p.transverse_to(B_p);

  v_E = E_p.cross(B_p) / B_p.squared();
  v_En = v_E.length();

  v_h = 0.5 * (v_nn + v_n);
  v_hp = v_h.parallel_to(B_p);
  v_ht = v_h.transverse_to(B_p);
  u = (v_ht - v_E).length();
  PetscFunctionReturn(PETSC_SUCCESS);
}


void RicketsonPush::adaptive_time_stepping(const Point& /* point */)
{
  PetscFunctionBeginUser;
  PetscReal B_norm = B_p.length();
  PetscReal Omega = q * B_norm / m;
  PetscReal rho = u / Omega;

  PetscReal delta_t = rho * (DB_pt.length() / B_norm);
  PetscReal delta_p = v_hp.length() / Omega * (DB_pp.length() / B_norm);
  PetscReal delta_E = v_En / Omega * (DB_pt.length() / B_norm);

  PetscReal Omega_dt = alpha *
    std::min(M_SQRT2 / std::sqrt(delta_E + delta_p),
      std::min(Omega * t_res, gamma / delta_t));

  // (eh) -- estimate of gyration velocity on half time-step.
  PetscReal u_eh =
    (v_ht - v_E).length() / std::sqrt(1.0 + 0.25 * POW2(Omega_dt));

  /// @todo Probably, some diagnostic is need here to understand the cases.
  if (v_En > (1.0 + beta) * u_eh) {
    dt = Omega_dt / Omega;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (v_En > (1.0 - beta) * u) {
    dt = std::min(t_res, gamma / Omega);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  Omega_dt = alpha *
    std::min(2.0 * std::sqrt(Omega * t_res / M_PI),
      2.0 * M_SQRT2 *
        std::min(1.0 / std::sqrt(delta_t), 1.0 / std::sqrt(delta_p)));

  dt = Omega_dt / Omega;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/**
 * @brief Evaluates nonlinear function F(x).
 * @param[in]  snes the SNES context.
 * @param[in]  vx   input vector of k-th iteration.
 * @param[in]  ctx  optional user-defined context.
 * @param[out] vf   function vector to be evaluated.
 *
 * @note `SNESNRICHARDSON` will iterate the following: x^{k+1} = x^{k} - lambda
 * F(x^{k}), where lambda -- damping coefficient, lambda = +1.0 by default (no
 * damping).
 */
void RicketsonPush::form_picard_iteration(
  SNES /* snes */, Vec vx, Vec vf, void* vctx)
{
  PetscFunctionBeginUser;
  auto& ctx = *reinterpret_cast<RicketsonPush::Context*>(vctx);
  const Vector3R& x_n = x_n;
  const Vector3R& v_n = v_n;

  const PetscReal* x;
  VecGetArrayRead(vx, &x);
  Vector3R x_nn{x[0], x[1], x[2]};
  Vector3R v_nn{x[3], x[4], x[5]};
  VecRestoreArrayRead(vx, &x);

  update(x_nn, v_nn);

  // To avoid the update of `Context::B_p`.
  Vector3R B_p = B_p;

  /// @todo Add guards in case of v_t << v_p
  PetscReal mu =
    -0.125 * m * ((v_nn - v_n).parallel_to(B_p)).squared() / B_p.length();
  Vector3R G_p = -mu * DB_pp;
  Vector3R G_t;

  if (u > v_En) {
    G_t = 2.0 * mu * DB_pt;
  }
  else {
    PetscReal zeta = 2.0 / POW2(u / v_En);
    PetscReal v_hpn = v_hp.length();
    PetscReal v_htn = v_ht.length();

    Vector3R v_Ed = v_E.normalized();
    Vector3R f_pE = DB_pt.parallel_to(v_Ed);
    Vector3R f_tE = DB_pt.transverse_to(v_Ed);

    G_t = zeta * v_hpn / v_htn * G_p.elementwise_product(v_Ed) -
      mu * (zeta * f_pE + f_tE / (1.0 - 1.0 / zeta));
  }

  B_p += (G_p + G_t).cross(v_ht) / (q * v_ht.squared());

  PetscReal tqm = 0.5 * dt * q / m;
  Vector3R a = v_n + tqm * E_p;

  // Velocity on a new half-step, v^{n+1/2, k+1}.
  Vector3R v_h = (a + tqm * a.cross(B_p) + POW2(tqm) * a.dot(B_p) * B_p) /
    (1.0 + POW2(tqm * B_p.length()));

  PetscReal* f;
  VecGetArrayWrite(vf, &f);
  f[0] = x_nn[X] - (x_n[X] + dt * v_h[X]);
  f[1] = x_nn[Y] - (x_n[Y] + dt * v_h[Y]);
  f[2] = x_nn[Z] - (x_n[Z] + dt * v_h[Z]);

  f[3] = v_nn[X] - (2.0 * v_h[X] - v_n[X]);
  f[4] = v_nn[Y] - (2.0 * v_h[Y] - v_n[Y]);
  f[5] = v_nn[Z] - (2.0 * v_h[Z] - v_n[Z]);
  VecRestoreArrayWrite(vf, &f);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
