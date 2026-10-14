#include "drift_kinetic_push.h"

DriftKineticPush::DriftKineticPush(PetscReal qm, PetscReal mp)
  : qm(qm), mp(mp)
{
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

void DriftKineticPush::process(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  converged = false;
  FRk = PETSC_MAX_REAL;
  FVk = PETSC_MAX_REAL;

  process_picard(dt, pn, p0);
}

void DriftKineticPush::process_picard(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  for (it = 0; it < maxit; it++) {
    evaluate_rhs(pn, p0);

    if (check_convergence(dt, pn, p0)) {
      converged = true;
      return;
    }

    update_r(dt, pn, p0);
    update_v_parallel(dt, pn, p0);
  }

  evaluate_rhs(pn, p0);
  converged = check_convergence(dt, pn, p0);
}

void DriftKineticPush::step(
  const PetscReal dt, PointByField& pn, const PointByField& p0)
{
  update_V_par(pn, p0);
  update_fields(pn, p0);
  update_V_dgc(pn, p0);
  update_a_par(pn, p0);
  update_r(dt, pn, p0);
  update_v_parallel(dt, pn, p0);
}

// Evaluates drift velocity and acceleration.
void DriftKineticPush::evaluate_rhs(const PointByField& pn, const PointByField& p0)
{
  update_V_par(pn, p0);
  update_fields(pn, p0);
  update_V_dgc(pn, p0);
  update_a_par(pn, p0);
}

void DriftKineticPush::update_fields(
  const PointByField& pn, const PointByField& p0)
{
  set_fields(p0.r, pn.r, E, lenB, b, gradB, rotB);

  rotB = (b.cross(gradB) + rotB) / lenB;

  B_eff = lenB * b + (V_par / qm) * rotB;
  lenB_eff = b.dot(B_eff);
  b_eff = B_eff / lenB_eff;

  F_eff = (qm * mp * E - p0.mu_p * gradB);
}

void DriftKineticPush::update_V_dgc(const PointByField& pn, const PointByField& p0)
{
  V_dgc = V_par * b_eff + F_eff.cross(b / (qm * mp * lenB_eff));
}

void DriftKineticPush::update_V_par(const PointByField& pn, const PointByField& p0)
{
  V_par = 0.5 * (pn.p_parallel + p0.p_parallel);
}

void DriftKineticPush::update_a_par(const PointByField& pn, const PointByField& p0)
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

bool DriftKineticPush::check_convergence(
  PetscReal dt, const PointByField& pn, const PointByField& p0)
{
  FRk = (pn.r - p0.r - dt * V_dgc).length();
  FVk = std::abs(pn.p_parallel - p0.p_parallel - dt * a_par);

  const PetscReal r_scale = atol + rtol * (dt * V_dgc).length();
  const PetscReal v_scale = atol + rtol * std::abs(dt * a_par);

  return (FRk <= r_scale) && (FVk <= v_scale);
}
