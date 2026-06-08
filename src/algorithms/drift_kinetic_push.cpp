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

void DriftKineticPush::set_fields_callback(SetFields&& callback)
{
  this->set_fields = std::move(callback);
}

void DriftKineticPush::process(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  for (it = 0; it < maxit; ++it) {
    step(dt, pn, p0);
    if (check_discrepancy(dt, pn, p0)) {
      converged = true;
      return;
    }
  }
  LOG("WARNING: DK Push failed to converge after {} iterations. FRk={: .4e}, FVhk={: .4e}",
      maxit, FRk, FVhk);
}

void DriftKineticPush::step(const PetscReal dt, PointByField& pn, const PointByField& p0) {
  update_Vh(pn, p0);
  update_fields(pn, p0);
  update_Vp(pn, p0);
  update_ah(pn, p0);
  update_r(dt, pn, p0);
  update_v_parallel(dt, pn, p0);
}

void DriftKineticPush::update_Vp(const PointByField& pn, const PointByField& p0) {
  Vp = Vh * bh_eff + F_eff.cross(bh / (qm * mp * lenBh));
}

void DriftKineticPush::update_Vh(const PointByField& pn, const PointByField& p0) {
  Vh = 0.5 * (pn.p_parallel + p0.p_parallel);
}

void DriftKineticPush::update_ah(const PointByField& pn, const PointByField& p0) {
  ah = F_eff.dot(bh_eff) / mp;
}

bool DriftKineticPush::check_discrepancy(PetscReal dt, const PointByField& pn, const PointByField& p0){
  FRk = get_residue_r(dt, pn, p0);
  FVhk = get_residue_v(dt, pn, p0);

  dRk = pn.r - p0.r;
  dVhk = pn.p_parallel - p0.p_parallel;

  return (FRk < atol) && //
         (FVhk * dt < atol);
}

PetscReal DriftKineticPush::get_residue_r(PetscReal dt, const PointByField& pn,
  const PointByField& p0)
{
  return (pn.r - p0.r - dRk).length();
}

PetscReal DriftKineticPush::get_residue_v(PetscReal dt, const PointByField& pn,
  const PointByField& p0)
{
  return std::abs((pn.p_parallel - p0.p_parallel) - dVhk);
}

void DriftKineticPush::update_r(PetscReal dt, PointByField& pn, const PointByField& p0)
{
  pn.r = p0.r + dt * Vp;
}

void DriftKineticPush::update_v_parallel(PetscReal dt, PointByField& pn, const PointByField& p0)
{
  pn.p_parallel = p0.p_parallel + dt * ah;
}

void DriftKineticPush::update_fields(const PointByField& pn, const PointByField& p0) {

  set_fields(p0.r, pn.r, Eh, Bh, gradBh, rotBh);

  bh = Bh.normalized();
  lenBh = Bh.length();

  rotbh = (bh.cross(gradBh) + rotBh) / lenBh;
  Bh_eff = Bh + (Vh / qm) * rotbh;
  lenBh_eff = bh.dot(Bh_eff);

  bh_eff = Bh_eff/lenBh_eff;
  F_eff = (qm*mp*Eh - p0.mu_p*gradBh);
}
