#include "drift_kinetic_push.h"

/// @file drift_kinetic_push.cpp
/// @brief Implementation of the fully implicit drift-kinetic guiding-center pusher.
/// @see drift_kinetic_push.h for the description of the algorithm and the mapping
/// of the members to the drift-kinetic equations of motion.

DriftKineticPush::DriftKineticPush(PetscReal qm, PetscReal mp)
  : qm(qm), mp(mp)
{
}

void DriftKineticPush::set_tolerances(
  PetscReal atol, PetscReal rtol, PetscInt maxit)
{
  this->atol = atol;
  this->rtol = rtol;
  this->maxit = maxit;
}

void DriftKineticPush::set_qm(PetscReal qm)
{
  this->qm = qm;
}

void DriftKineticPush::set_mp(PetscReal mp)
{
  this->mp = mp;
}

PetscReal DriftKineticPush::get_mp() const
{
  return this->mp;
}

PetscReal DriftKineticPush::get_qm() const
{
  return this->qm;
}

PetscInt DriftKineticPush::get_iteration_number() const
{
  return it;
}

bool DriftKineticPush::has_converged() const
{
  return converged;
}

void DriftKineticPush::set_fields_callback(SetFields&& callback)
{
  this->set_fields = std::move(callback);
}

void DriftKineticPush::set_solver(DriftKineticSolver solver)
{
  this->solver = solver;
}

void DriftKineticPush::set_relaxation(PetscReal beta0, PetscReal beta_min)
{
  this->beta0 = beta0;
  this->beta_min = beta_min;
}

void DriftKineticPush::set_anderson_clip(PetscReal gamma_min, PetscReal gamma_max)
{
  this->gamma_min = gamma_min;
  this->gamma_max = gamma_max;
}

void DriftKineticPush::set_divergence_factor(PetscReal div_factor)
{
  this->div_factor = div_factor;
}

PetscReal DriftKineticPush::get_residual_norm() const
{
  return fnorm;
}

/// @brief Solves the midpoint equations of motion as a fixed-point problem.
/// Dispatches to the selected nonlinear solver and reports non-convergence.
void DriftKineticPush::process(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  converged = false;
  FRk = FVhk = PETSC_MAX_REAL;

  switch (solver) {
    case DriftKineticSolver::DKPICARD:
      process_picard(dt, pn, p0);
      break;
    case DriftKineticSolver::DKANDERSON:
      process_anderson(dt, pn, p0);
      break;
  }

  if (!converged)
    LOG("WARNING: DK Push not converged: it={}, |f|={: .3e}, FRk={: .4e}, FVhk={: .4e}",
        it, fnorm, FRk, FVhk);
}

/// @brief Fixed-point solve by a safeguarded, adaptively damped Anderson(1)
/// acceleration over the Picard map G.
/// The map and its residual:  G(x) = (R^n + dt*Vp(x),  v^n + dt*ah(x)),
///                            f(x) = G(x) - x,  with x = (R^{n+1}, v_par^{n+1}).
void DriftKineticPush::process_anderson(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  AndersonState st;
  st.beta = beta0;

  for (it = 0; it < maxit; ++it) {
    // Evaluate the Picard map: iterate x, image g, residual f and its norm ‖f‖.
    evaluate_map(dt, pn, p0, st);

    // Check the true nonlinear residual before taking the next step.
    if (check_discrepancy(dt, pn, p0)) { converged = true; return; }

    // Hand control back to the outer substep logic if the iteration diverges.
    if (is_diverging(st)) return;

    // Safeguarded Anderson(1) (or damped-Picard fallback) update of the iterate.
    anderson_update(pn, st);

    // Remember the current iterate/image/residual for the next acceleration step.
    shift_history(st);
  }

  // Final check after the last update.
  evaluate_rhs(pn, p0);
  converged = check_discrepancy(dt, pn, p0);
}

/// @brief Forms the Picard map image `g = (R^n + dt*Vp, v^n + dt*ah)`, the
/// residual `f = g - x` and its norm at the current iterate `pn`.
void DriftKineticPush::evaluate_map(
  PetscReal dt, const PointByField& pn, const PointByField& p0, AndersonState& st)
{
  // Current iterate x_k = (pn.r, pn.p_parallel).
  st.x_r = pn.r;
  st.x_v = pn.p_parallel;

  // G(x_k): evaluate RHS (sets Vp, ah), then form image g_k and residual f_k.
  evaluate_rhs(pn, p0);
  st.g_r = p0.r + dt * Vp;
  st.g_v = p0.p_parallel + dt * ah;
  st.f_r = st.g_r - st.x_r;
  st.f_v = st.g_v - st.x_v;

  fnorm = std::sqrt(st.f_r.dot(st.f_r) + st.f_v * st.f_v);
  if (st.fnorm0 < 0.0) st.fnorm0 = fnorm;
}

/// @brief Divergence guard: the residual is non-finite or has grown beyond
/// `div_factor` times its initial value, so the outer substep control takes over.
bool DriftKineticPush::is_diverging(const AndersonState& st) const
{
  return !std::isfinite(fnorm) || fnorm > div_factor * st.fnorm0;
}

/// @brief Safeguarded Anderson(1) acceleration of the iterate. Accelerates only
/// from a consecutive pair whose residual did not grow; otherwise falls back to
/// a damped-Picard step that shrinks `beta` to recondition the next pair.
void DriftKineticPush::anderson_update(PointByField& pn, AndersonState& st)
{
  const bool accelerate = st.have_history && fnorm <= st.fnorm_prev;
  if (accelerate) {
    // Anderson(1) extrapolation coefficient from the residual increment.
    const Vector3R  dfr = st.f_r - st.fr_prev;
    const PetscReal dfv = st.f_v - st.fv_prev;
    const PetscReal den = dfr.dot(dfr) + dfv * dfv;
    PetscReal gamma = (den > 1e-300) ? (st.f_r.dot(dfr) + st.f_v * dfv) / den : 0.0;
    gamma = std::clamp(gamma, gamma_min, gamma_max);

    // Relaxed Anderson(1):  x_{k+1} = x̄ + beta (ḡ - x̄)
    //   x̄ = x_k - gamma (x_k - x_{k-1}),  ḡ = g_k - gamma (g_k - g_{k-1})
    const Vector3R  xbar_r = st.x_r - gamma * (st.x_r - st.xr_prev);
    const PetscReal xbar_v = st.x_v - gamma * (st.x_v - st.xv_prev);
    const Vector3R  gbar_r = st.g_r - gamma * (st.g_r - st.gr_prev);
    const PetscReal gbar_v = st.g_v - gamma * (st.g_v - st.gv_prev);

    pn.r          = xbar_r + st.beta * (gbar_r - xbar_r);
    pn.p_parallel = xbar_v + st.beta * (gbar_v - xbar_v);

    st.beta = std::min(1.0, st.beta * 1.3);  // residual is dropping: relax the damping
  } else {
    // Damped Picard fallback (also the it==0 step). Restart-friendly: a bad
    // step shrinks beta so the next consecutive pair is well-conditioned.
    pn.r          = st.x_r + st.beta * st.f_r;
    pn.p_parallel = st.x_v + st.beta * st.f_v;
    if (st.have_history && fnorm > st.fnorm_prev)
      st.beta = std::max(beta_min, 0.5 * st.beta);
  }
}

/// @brief Shifts the current iterate/image/residual into the history slots used
/// by the next Anderson(1) acceleration step.
void DriftKineticPush::shift_history(AndersonState& st)
{
  st.xr_prev = st.x_r; st.xv_prev = st.x_v;
  st.gr_prev = st.g_r; st.gv_prev = st.g_v;
  st.fr_prev = st.f_r; st.fv_prev = st.f_v;
  st.fnorm_prev = fnorm;
  st.have_history = true;
}

/// @brief Classical Picard iteration over the nonlinear midpoint equations until
/// the position and parallel-momentum residuals fall below the tolerances.
void DriftKineticPush::process_picard(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  for (it = 0; it < maxit; ++it) {
    // Evaluate RHS at the current nonlinear guess pn (sets Vp, ah).
    evaluate_rhs(pn, p0);

    // Check the true nonlinear residual before taking the next step.
    if (check_discrepancy(dt, pn, p0)) { converged = true; return; }

    // Picard update of position and parallel momentum.
    update_r(dt, pn, p0);
    update_v_parallel(dt, pn, p0);
  }

  // Final check after the last Picard update.
  evaluate_rhs(pn, p0);
  converged = check_discrepancy(dt, pn, p0);
}

/// @brief Single Picard sweep: recompute the midpoint state, then update the
/// guiding-center position and parallel momentum from it.
void DriftKineticPush::step(const PetscReal dt, PointByField& pn, const PointByField& p0) {
  update_Vh(pn, p0);
  update_fields(pn, p0);
  update_Vp(pn, p0);
  update_ah(pn, p0);
  update_r(dt, pn, p0);
  update_v_parallel(dt, pn, p0);
}

void DriftKineticPush::evaluate_rhs(
  const PointByField& pn,
  const PointByField& p0)
{
  update_Vh(pn, p0);
  update_fields(pn, p0);
  update_Vp(pn, p0);
  update_ah(pn, p0);
}

/// @brief Drift-velocity equation: parallel streaming along the effective field
/// plus the @f$(q\mathbf{E}-\mu\nabla B)\times\mathbf{b}/(qB)@f$ drift.
void DriftKineticPush::update_Vp(const PointByField& pn, const PointByField& p0) {
  Vp = Vh * bh_eff + F_eff.cross(bh / (qm * mp * lenBh_eff));
}

/// @brief Midpoint parallel velocity @f$v_\parallel^{n+1/2}=(v^{n+1}+v^{n})/2@f$.
void DriftKineticPush::update_Vh(const PointByField& pn, const PointByField& p0) {
  Vh = 0.5 * (pn.p_parallel + p0.p_parallel);
}

/// @brief Parallel-momentum equation: effective force projected on the effective field.
void DriftKineticPush::update_ah(const PointByField& pn, const PointByField& p0) {
  ah = F_eff.dot(bh_eff) / mp;
}

bool DriftKineticPush::check_discrepancy(PetscReal dt, const PointByField& pn, const PointByField& p0){
  FRk = get_residue_r(dt, pn, p0);
  FVhk = get_residue_v(dt, pn, p0);

  // The residual of each implicit equation is measured against the size of the
  // update that equation produces over the step: the drift displacement
  // `dt * Vp` for the position and the parallel-velocity increment `dt * ah`.
  // This makes the test a bound on the *relative* error of the discrete
  // equations of motion, independent of the (arbitrary) coordinate origin and
  // of the absolute particle position. The relative scale is what carries the
  // tolerance; `atol` is only a floor guarding the degenerate case of a
  // vanishing update (Vp -> 0 or ah -> 0).
  const PetscReal r_scale = atol + rtol * (dt * Vp).length();
  const PetscReal v_scale = atol + rtol * std::abs(dt * ah);

  return (FRk <= r_scale) && (FVhk <= v_scale);
}

PetscReal DriftKineticPush::get_residue_r( PetscReal dt, const PointByField& pn, const PointByField& p0)
{
  return (pn.r - p0.r - dt * Vp).length();
}

PetscReal DriftKineticPush::get_residue_v(PetscReal dt, const PointByField& pn,
  const PointByField& p0)
{
  return std::abs(pn.p_parallel - p0.p_parallel - dt * ah);
}

/// @brief Implicit position update @f$\mathbf{R}^{n+1}=\mathbf{R}^{n}+\tau\mathbf{V}_p@f$.
void DriftKineticPush::update_r(PetscReal dt, PointByField& pn, const PointByField& p0)
{
  pn.r = p0.r + dt * Vp;
}

/// @brief Implicit parallel-momentum update @f$v^{n+1}=v^{n}+\tau\,a@f$.
void DriftKineticPush::update_v_parallel(PetscReal dt, PointByField& pn, const PointByField& p0)
{
  pn.p_parallel = p0.p_parallel + dt * ah;
}

/// @brief Refreshes the midpoint state: interpolates the fields to the particle,
/// then forms the curl of the unit field, the effective field @f$\mathbf{B}^{*}@f$
/// and the effective force @f$q\mathbf{E}-\mu\nabla B@f$ used by the motion equations.
void DriftKineticPush::update_fields(const PointByField& pn, const PointByField& p0) {
  // Interpolated fields at the midpoint of the trajectory.
  set_fields(p0.r, pn.r, Eh, lenBh, bh, gradBh, rotBh);

  // Derived quantities.
  bh.normalized();
  rotbh = (bh.cross(gradBh) + rotBh) / lenBh;

  Bh_eff = lenBh * bh + (Vh / qm) * rotbh;
  lenBh_eff = bh.dot(Bh_eff);
  bh_eff = Bh_eff / lenBh_eff;

  F_eff = (qm * mp * Eh - p0.mu_p * gradBh);
}
