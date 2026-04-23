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
  return converged_;
}

void DriftKineticPush::set_fields_callback(SetFields&& callback)
{
  this->set_fields = std::move(callback);
}

void DriftKineticPush::set_B_callback(SetFields&& callback)
{
  this->set_B = std::move(callback);
}

void DriftKineticPush::process(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  pre_step(dt, pn, p0);
  for (it = 0; it < maxit; ++it) {
    update_Vp(pn, p0);
    if (check_discrepancy(dt, pn, p0)) {
      last_step(dt, pn, p0);
      converged_ = true;
      return;
    }
    step(dt, pn, p0);
  }
  LOG("WARNING: DK Push failed to converge after {} iterations. Rn={: .4e}, Vn={: .4e}",
      maxit, Rn, Vn);
  last_step(dt, pn, p0);
}

void DriftKineticPush::pre_step(const PetscReal dt, PointByField& pn, const PointByField& p0) {
  update_fields(pn, p0);
  update_Vp(pn, p0);
  R0 = get_residue_r(dt, pn, p0);
  V0 = get_residue_v(dt, pn, p0);
}

void DriftKineticPush::step(const PetscReal dt, PointByField& pn, const PointByField& p0) {
  update_r(dt, pn, p0);
  update_fields(pn, p0);
  update_v_parallel(dt, pn, p0);
}

void DriftKineticPush::last_step(const PetscReal dt, PointByField& pn, const PointByField& p0) {
  step(dt, pn, p0);
  update_v_perp(pn, p0);
}

void DriftKineticPush::update_Vp(const PointByField& pn, const PointByField& p0) {
  Vh = 0.5 * (pn.p_parallel + p0.p_parallel);
  Vp = Vh * bh + get_Vd(p0);
}

Vector3R DriftKineticPush::get_Vd(const PointByField& p0)
{
  return (Vh * Vh / lenBh + p0.mu_p / mp) * bh.cross(gradBh / lenBh) / qm + Eh.cross(bh) / lenBh;
}

bool DriftKineticPush::check_discrepancy(PetscReal dt, const PointByField& pn, const PointByField& p0){
  Rn = get_residue_r(dt, pn, p0);
  Vn = get_residue_v(dt, pn, p0);
  return (Rn < atol + rtol * R0) && (Vn < atol + rtol * V0);
}

PetscReal DriftKineticPush::get_residue_r(PetscReal dt, const PointByField& pn,
  const PointByField& p0)
{
  return (pn.r - p0.r - dt * Vp).length();
}

PetscReal DriftKineticPush::get_residue_v(PetscReal dt, const PointByField& pn,
  const PointByField& p0)
{
  return std::abs((pn.p_parallel - p0.p_parallel) - dt * (get_v_parallel(p0) + get_F(dt, pn, p0) / mp));
}

void DriftKineticPush::update_r(PetscReal dt, PointByField& pn, const PointByField& p0)
{
  pn.r = p0.r + dt * Vp;
}

void DriftKineticPush::update_v_perp(PointByField& pn, const PointByField& p0)
{
  set_B(p0.r, pn.r, B0, meanB, Bn);
  pn.p_perp = std::sqrt(2.*pn.mu_p*Bn.length() / mp);
}

void DriftKineticPush::update_v_parallel(PetscReal dt, PointByField& pn, const PointByField& p0)
{
  pn.p_parallel = p0.p_parallel + dt * (get_v_parallel(p0) + get_F(dt, pn, p0) / mp);
}

PetscReal DriftKineticPush::get_F(PetscReal dt, const PointByField& pn, const PointByField& p0) {
  PetscReal eps = 1e-10;
  PetscReal inv_Vh = Vh / (Vh * Vh + eps * eps);

  PetscReal F = - p0.mu_p * (bn - b0).dot(meanB) * inv_Vh / dt;
  return F;
}

PetscReal DriftKineticPush::get_v_parallel(const PointByField& p0) {
  Vector3R bracket = qm * Eh - p0.mu_p / mp * gradBh;
  Vector3R ratio = bh + (1.0 / qm) * Vh / lenBh * bh.cross(gradBh / lenBh);
  return bracket.dot(ratio);
}

void DriftKineticPush::update_fields(const PointByField& pn, const PointByField& p0) {
  set_fields(p0.r, pn.r, Eh, Bh, gradBh);
  set_B(p0.r, pn.r, B0, meanB, Bn);
  b0 = B0.normalized();
  bh = Bh.normalized();
  bn = Bn.normalized();
  lenBh = Bh.length();
}
