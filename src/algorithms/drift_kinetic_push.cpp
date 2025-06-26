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

PetscInt DriftKineticPush::get_iteration_number() const
{
  return it;
}

void DriftKineticPush::set_fields_callback(SetFields&& callback)
{
  this->set_fields = std::move(callback);
}


void DriftKineticPush::process(PetscReal dt, PointByField& pn, const PointByField& p0)
{
  PetscAssertAbort((bool)set_fields, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "DriftKineticPush::set_fields have to be specified");

  Vector3R Vd = {0.,0.,0.};
  PetscReal Vh = 0.;

  PetscReal R1 = 0., R2 = 0.;

  set_fields(p0.r, Ep, Bp, gradBp);

  Vector3R Eh = {0.,0.,0.}, E0 = Ep;

  Vector3R B0 = Bp, Bh = Bp;
  Vector3R gradB0 = gradBp, gradBh = gradBp;
  Vector3R b0 = Bp.normalized();
  Vector3R bp = b0, h = b0;

  for (it = 0; it < maxit; ++it) {
    Vh = 0.5 * (pn.p_parallel + p0.p_parallel);
    Vd = get_Vd(p0, h, Vh, Bh.length(), gradBh, Eh);

    R1 = get_residue_r(dt, pn, p0, Vh*h, Vd);
    R2 = get_residue_v(dt, pn, p0, Vh, h, Vd, B0, Eh);
    if ((R1 < eps) & (R2 < delta))
      return;

    update_r(dt, pn, p0, Vh*h, Vd);
    update_v_perp(pn, p0, B0);
    update_v_parallel(dt, pn, p0, Vh, h, Vd, B0, Eh);

    set_fields(pn.r, Ep, Bp, gradBp);
    Eh = 0.5 * (Ep + E0);
    Bh = 0.5 * (Bp + B0);
    gradBh = 0.5 * (gradBp + gradB0);
    bp = Bp.normalized();
    h = 0.5 * (bp + b0);

  }

  PetscCheckAbort((R1 >= eps) || (R2 >= delta), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "DriftKineticPush::process() nonlinear iterations diverged with norm %e and %e!", R1, R2);
}


Vector3R DriftKineticPush::get_Vd(const PointByField& p0, const Vector3R& h, const PetscReal& Vh, const PetscReal& Bh, const Vector3R& gradBh, const Vector3R& Eh) const
{
  return Eh.cross(h)/Bp.length()
          + 1./qm * (Vh*Vh/Bh + p0.mu_p/mp) * h.cross(gradBh/Bh);
}

void DriftKineticPush::update_r(PetscReal dt, PointByField& pn, const PointByField& p0, const Vector3R& Vh, const Vector3R& Vd) const {
  pn.r = p0.r + dt*(Vh + Vd);
}

void DriftKineticPush::update_v_perp(PointByField& pn, const PointByField& p0, const Vector3R& B0) const{
  pn.p_perp = p0.p_perp * std::sqrt(Bp.length()/B0.length());
}

void DriftKineticPush::update_v_parallel(PetscReal dt, PointByField& pn, const PointByField& p0, const PetscReal& Vh, const Vector3R& h, const Vector3R& Vd, const Vector3R& B0, const Vector3R& Eh) const {
  pn.p_parallel = p0.p_parallel
                + dt*qm*(Eh.dot(h) + Eh.dot(Vd)/Vh) 
                - p0.mu_p/mp*(Bp.length() - B0.length())/Vh;
}

PetscReal DriftKineticPush::get_residue_r(PetscReal dt, const PointByField& pn, const PointByField& p0, const Vector3R& Vh, const Vector3R& Vd) const {
  return (pn.r - p0.r - dt*(Vh + Vd)).length();
}

PetscReal DriftKineticPush::get_residue_v(PetscReal dt, const PointByField& pn, const PointByField& p0, const PetscReal& Vh, const Vector3R& h, const Vector3R& Vd, const Vector3R& B0, const Vector3R& Eh) const {
  return std::abs((pn.p_parallel - p0.p_parallel) 
                  - dt*qm*(Eh.dot(h) + Eh.dot(Vd)/Vh) 
                  + p0.mu_p/mp*(Bp.length() - B0.length())/Vh);
}