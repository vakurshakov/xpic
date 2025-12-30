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

    //std::cout << it;

    if (check_discrepancy(dt, pn, p0) && it) {
      update_v_perp(pn, p0);
      //std::cout << it << ":" << Eh << " " << Bh << " " << B0 << " " << Bn <<std::endl;
      //std::cout << " " << gradBh << " " << pn.r << " "<< p0.r <<std::endl; 
      //std::cin >> it;
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
  //std::cout << R1 << " " << R2 << std::endl;
  return (R1 < eps) && (R2 < delta);
}

PetscReal DriftKineticPush::get_residue_r(PetscReal dt, const PointByField& pn,
  const PointByField& p0) const
{
  return (pn.r - p0.r - dt * Vp).length()/pn.r.length();
}

PetscReal DriftKineticPush::get_residue_v(PetscReal dt, const PointByField& pn,
  const PointByField& p0) const
{
  return std::abs((pn.p_parallel - p0.p_parallel) - dt * get_v_parallel(p0))/std::abs(pn.p_parallel);
}

void DriftKineticPush::update_r(PetscReal dt, PointByField& pn, const PointByField& p0) const
{
  pn.r = p0.r + dt * Vp;
}

void DriftKineticPush::update_v_perp(PointByField& pn, const PointByField& p0)
{
  pn.p_perp = p0.p_perp * std::sqrt(Bn.length() / B0.length());
}

void DriftKineticPush::update_v_parallel(PetscReal dt, PointByField& pn, const PointByField& p0) const
{
  pn.p_parallel = p0.p_parallel + dt * get_v_parallel(p0) + get_F(p0) / mp;
}

PetscReal DriftKineticPush::get_F(const PointByField& p0) const {
  return (std::abs(Vh) < 1e-12) ?  0. : - p0.mu_p * (bn - b0).dot(meanB) / Vh;
}

PetscReal DriftKineticPush::get_v_parallel(const PointByField& p0) const {
  PetscReal qm_term = (std::abs(Vh) < 1e-12) ? Eh.dot(bh) : (Eh.dot(Vp) / Vh);
  PetscReal mu_term = (std::abs(Vh) < 1e-12) ? gradBh.dot(bh) : (gradBh.dot(Vp) / Vh);
  return qm * qm_term - (p0.mu_p / mp) * mu_term;
}

void DriftKineticPush::update_fields(const PointByField& pn, const PointByField& p0) {
  Vector3R E0, gradB0;
  Vector3R En, gradBn;
  set_fields(p0.r, p0.r, E0, B0, gradB0);
  set_fields(p0.r, pn.r, En, Bn, gradBn);
  set_fields(p0.r, 0.5 * (p0.r + pn.r), Eh, Bh, gradBh);
  meanB = 0.5 * (Bn + B0);
  bh = Bh.normalized(), bn = Bn.normalized(), b0 = B0.normalized();
  lenBh = Bh.length();
}
