#include "drift_kinetic_push.h"

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

PetscReal DriftKineticPush::get_FRk() const
{
  return FRk;
}

PetscReal DriftKineticPush::get_FVk() const
{
  return FVk;
}

void DriftKineticPush::set_fields_callback(SetFields&& callback)
{
  this->set_fields = std::move(callback);
}

PetscErrorCode DriftKineticPush::process(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  PetscFunctionBeginHot;
  converged = false;
  FRk = PETSC_MAX_REAL;
  FVk = PETSC_MAX_REAL;

#if DRIFT_KINETIC_AITKEN == 0
  PetscCall(process_picard(dt, pn, p0));
#elif DRIFT_KINETIC_AITKEN == 1
  PetscCall(process_picard_aitken(dt, pn, p0));
#else
  #error "DRIFT_KINETIC_AITKEN must be 0 or 1"
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticPush::process_picard(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  PetscFunctionBeginHot;
  has_prev = false;

  for (it = 0; it < maxit; it++) {
    PetscCall(evaluate_rhs(pn, p0));

    if (check_convergence(dt, pn, p0)) {
      converged = true;
      PetscFunctionReturn(PETSC_SUCCESS);
    }

    update_r(dt, pn, p0);
    update_v_parallel(dt, pn, p0);

    V_dgc_prev = V_dgc;
    a_par_prev = a_par;
    has_prev = true;
  }

  PetscCall(evaluate_rhs(pn, p0));
  converged = check_convergence(dt, pn, p0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Aitken uses the sequential map increment; acceptance and convergence use
// the original equations. Rejected trials reduce omega from the same point.
PetscErrorCode DriftKineticPush::process_picard_aitken(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  PetscFunctionBeginHot;
  Vector3R dr = pn.r - p0.r;
  PetscReal dv = pn.p_parallel - p0.p_parallel;
  State previous{};
  PetscReal omega = 1;
  bool has_history = false;
  it = 0;
  PetscCall(evaluate_rhs(pn, p0));

  for (;;) {
    const Vector3R rr = dt * V_dgc - dr;
    const State residual{rr[X], rr[Y], rr[Z], dt * (dt * a_par - dv)};
    converged = check_increment_convergence(dt, residual);
    if (converged || it >= 4 * maxit)
      PetscFunctionReturn(PETSC_SUCCESS);

    PetscReal norm2 = 0;
    for (PetscReal value : residual)
      norm2 += value * value;
    if (!std::isfinite(norm2))
      PetscFunctionReturn(PETSC_SUCCESS);

    const PetscReal dv_next = dt * a_par;
    V_par = 0.5 * (p0.p_parallel + (p0.p_parallel + dv_next));
    update_effective_field();
    update_V_dgc();
    const Vector3R next = dt * V_dgc - dr;
    const State f{next[X], next[Y], next[Z], dt * (dv_next - dv)};

    if (has_history) {
      PetscReal numerator = 0, denominator = 0, scale = 0;
      for (PetscInt i = 0; i < 4; i++) {
        const PetscReal difference = f[i] - previous[i];
        numerator += previous[i] * difference;
        denominator += difference * difference;
        scale += f[i] * f[i] + previous[i] * previous[i];
      }
      if (denominator > PETSC_MACHINE_EPSILON * scale) {
        const PetscReal candidate = -omega * numerator / denominator;
        if (std::isfinite(candidate))
          omega = std::clamp(candidate, 0.05, 1.0);
      }
    }

    const PointByField current(pn);
    for (;;) {
      const Vector3R trial_dr = dr + omega * Vector3R(f.data());
      const PetscReal trial_dv = dv + omega * f[3] / dt;
      pn.r = p0.r + trial_dr;
      pn.p_parallel = p0.p_parallel + trial_dv;
      it++;
      PetscReal trial_norm2 = PETSC_MAX_REAL;
      if (std::isfinite(pn.p_parallel) && std::isfinite(pn.r[X]) &&
        std::isfinite(pn.r[Y]) && std::isfinite(pn.r[Z])) {
        PetscCall(evaluate_rhs(pn, p0));
        const Vector3R trial_rr = dt * V_dgc - trial_dr;
        const PetscReal trial_rv = dt * (dt * a_par - trial_dv);
        trial_norm2 = trial_rr.squared() + trial_rv * trial_rv;
        const State trial_residual{
          trial_rr[X], trial_rr[Y], trial_rr[Z], trial_rv};
        converged = check_increment_convergence(dt, trial_residual);
      }
      if (std::isfinite(trial_norm2) && (converged || trial_norm2 <= 4 * norm2)) {
        dr = trial_dr;
        dv = trial_dv;
        previous = f;
        has_history = true;
        break;
      }
      if (omega <= 0.05 || it >= 4 * maxit) {
        pn = current;
        PetscCall(evaluate_rhs(pn, p0));
        const Vector3R restore_rr = dt * V_dgc - dr;
        const State restore_residual{
          restore_rr[X], restore_rr[Y], restore_rr[Z], dt * (dt * a_par - dv)};
        converged = check_increment_convergence(dt, restore_residual);
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      omega = std::max(0.05, 0.5 * omega);
    }
  }
}

// Evaluates drift velocity and acceleration.
PetscErrorCode DriftKineticPush::evaluate_rhs(
  const PointByField& pn, const PointByField& p0)
{
  PetscFunctionBeginHot;
  update_V_par(pn, p0);
  PetscCall(update_fields(pn, p0));
  update_V_dgc();
  update_a_par();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticPush::update_fields(
  const PointByField& pn, const PointByField& p0)
{
  PetscFunctionBeginHot;
  PetscCall(set_fields(p0.r, pn.r, E, lenB, b, gradB, rotB));

  rotB = (b.cross(gradB) + rotB) / lenB;

  update_effective_field();

  F_eff = qm * mp * E - p0.mu_p * gradB;
  PetscFunctionReturn(PETSC_SUCCESS);
}

void DriftKineticPush::update_effective_field()
{
  B_eff = lenB * b + (V_par / qm) * rotB;
  lenB_eff = b.dot(B_eff);
  b_eff = B_eff / lenB_eff;
}

void DriftKineticPush::update_V_dgc()
{
  V_dgc = V_par * b_eff + F_eff.cross(b / (qm * mp * lenB_eff));
}

void DriftKineticPush::update_V_par(const PointByField& pn, const PointByField& p0)
{
  V_par = 0.5 * (pn.p_parallel + p0.p_parallel);
}

void DriftKineticPush::update_a_par()
{
  a_par = F_eff.dot(b_eff) / mp;
}

void DriftKineticPush::update_r(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  pn.r = p0.r + dt * V_dgc;
}

void DriftKineticPush::update_v_parallel(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  pn.p_parallel = p0.p_parallel + dt * a_par;
}

// After an update `pn.r - p0.r = dt * V_dgc_prev` up to rounding, so the
// residual is the Picard increment. It is computed without subtracting the
// absolute coordinates, whose rounding error eps * |r| would otherwise set
// the floor of `FRk`. The warm start (`pn` from the previous outer iteration)
// has no stored right-hand side and is checked by the absolute coordinates.
bool DriftKineticPush::check_convergence(
  PetscReal dt, const PointByField& pn, const PointByField& p0)
{
  if (has_prev) {
    FRk = dt * (V_dgc_prev - V_dgc).length();
    FVk = dt * std::abs(a_par_prev - a_par);
  }
  else {
    FRk = (pn.r - p0.r - dt * V_dgc).length();
    FVk = std::abs(pn.p_parallel - p0.p_parallel - dt * a_par);
  }

  const PetscReal r_scale = dt * atol + rtol * (dt * V_dgc).length();
  const PetscReal v_scale =
    atol + rtol * (std::abs(V_par) + std::abs(dt * a_par));

  return (FRk <= r_scale) && (FVk <= v_scale);
}

bool DriftKineticPush::check_increment_convergence(PetscReal dt, const State& f)
{
  FRk = Vector3R(f.data()).length();
  FVk = std::abs(f[3]) / dt;

  const PetscReal r_scale = dt * atol + rtol * (dt * V_dgc).length();
  const PetscReal v_scale =
    atol + rtol * (std::abs(V_par) + std::abs(dt * a_par));

  return (FRk <= r_scale) && (FVk <= v_scale);
}
