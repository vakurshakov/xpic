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

  PetscCall(process_picard(dt, pn, p0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Sequential Picard: the parallel velocity is updated first, then the drift
// velocity with the new `V_par` (fields at the previous position) moves `r`.
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

    update_v_parallel(dt, pn, p0);
    update_V_par(pn, p0);
    update_effective_field();
    update_V_dgc();
    update_r(dt, pn, p0);

    V_dgc_prev = V_dgc;
    a_par_prev = a_par;
    has_prev = true;
  }

  PetscCall(evaluate_rhs(pn, p0));
  converged = check_convergence(dt, pn, p0);
  PetscFunctionReturn(PETSC_SUCCESS);
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
