#include "drift_kinetic_push.h"

DriftKineticPush::DriftKineticPush(PetscReal qm, PetscReal mp)
  : qm(qm), mp(mp)
{
}

void DriftKineticPush::set_tolerances(
  PetscReal eps, PetscReal delta, PetscInt maxit)
{
  this->eps = eps;
  this->delta = delta;
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

void DriftKineticPush::set_fields_callback(SetFields&& callback)
{
  this->set_fields = std::move(callback);
}


void DriftKineticPush::process(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  pre_step(pn, p0);
  
#if 0
  LOG("=== DriftKineticPush::process START ===");
  LOG("dt = {}", dt);
  LOG("p0.r = {}", p0.r);
  LOG("p0.p_parallel = {}", p0.p_parallel);
  LOG("p0.p_perp = {}", p0.p_perp);
  LOG("p0.mu_p = {}", p0.mu_p);
  LOG("B0 = {}, |B0| = {}", Bh, Bh.length());
  LOG("mp = {}, qm = {}", mp, qm);
  LOG("Eh = {}", Eh);
#endif

  PetscAssertAbort((bool)set_fields, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "DriftKineticPush::set_fields have to be specified");


  for (it = 0; it < maxit; ++it) {
    update_Vp(pn, p0);

    if (check_discrepancy(dt, pn, p0) && it) {
      update_v_perp(pn, p0);
      return;
    }

    update_r(dt, pn, p0);
    update_fields(pn, p0);
    update_v_parallel(dt, pn, p0);
  }

  PetscCheckAbort((R1 >= eps) || (R2 >= delta), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "DriftKineticPush::process() nonlinear iterations diverged with norm %e and %e!", R1, R2);
}

void DriftKineticPush::pre_step(const PointByField& pn, const PointByField& p0) {
  update_fields(pn, p0);
  R1 = 0.;
  R2 = 0.;
}

void DriftKineticPush::update_Vp(const PointByField& pn, const PointByField& p0) {
  Vh = 0.5 * (pn.p_parallel + p0.p_parallel);
  Vp = Vh * bh + get_Vd(p0);
}

Vector3R DriftKineticPush::get_Vd(const PointByField& p0) const
{
  return (Vh * Vh / lenBh + p0.mu_p / mp) * bh.cross(gradBh / lenBh) / qm + Eh.cross(bh) / lenBh;
}

bool DriftKineticPush::check_discrepancy(PetscReal dt, const PointByField& pn, const PointByField& p0){
  R1 = get_residue_r(dt, pn, p0);
  R2 = get_residue_v(dt, pn, p0);
  return (R1 < eps) && (R2 < delta);
}

PetscReal DriftKineticPush::get_residue_r(PetscReal dt, const PointByField& pn,
  const PointByField& p0) const
{
  return (pn.r - p0.r - dt * Vp).length();
}

PetscReal DriftKineticPush::get_residue_v(PetscReal dt, const PointByField& pn,
  const PointByField& p0) const
{
  return std::abs((pn.p_parallel - p0.p_parallel) - dt * get_v_parallel(p0));
}

void DriftKineticPush::update_r(PetscReal dt, PointByField& pn, const PointByField& p0) const
{
  pn.r = p0.r + dt * Vp;
}

void DriftKineticPush::update_v_perp(PointByField& pn, const PointByField& p0)
{
  Vector3R B0, Bn;
  DriftKineticPush::set_fields(p0.r, p0.r, Eh, B0, gradBh);
  DriftKineticPush::set_fields(p0.r, pn.r, Eh, Bn, gradBh);
  pn.p_perp = p0.p_perp * std::sqrt(Bn.length() / B0.length());
}

void DriftKineticPush::update_v_parallel(PetscReal dt, PointByField& pn, const PointByField& p0) const
{
  pn.p_parallel = p0.p_parallel + dt * get_v_parallel(p0);
}

PetscReal DriftKineticPush::get_v_parallel(const PointByField& p0) const {
  PetscReal qm_term = (std::abs(Vh) < 1e-12) ? Eh.dot(bh) : (Eh.dot(Vp) / Vh);
  PetscReal mu_term = (std::abs(Vh) < 1e-12) ? gradBh.dot(bh) : (gradBh.dot(Vp) / Vh);
  return qm * qm_term - (p0.mu_p / mp) * mu_term;
}

void DriftKineticPush::update_fields(const PointByField& pn, const PointByField& p0) {
  set_fields(p0.r, (p0.r + pn.r)/2., Eh, Bh, gradBh);
  bh = Bh.normalized();
  lenBh = Bh.length();
}
