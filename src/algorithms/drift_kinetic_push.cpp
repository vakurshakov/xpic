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
  set_fields(p0.r, pn.r, Eh, Bp, gradBp);

#if 0
  LOG("=== DriftKineticPush::process START ===");
  LOG("dt = {}", dt);
  LOG("p0.r = {}", p0.r);
  LOG("p0.p_parallel = {}", p0.p_parallel);
  LOG("p0.p_perp = {}", p0.p_perp);
  LOG("p0.mu_p = {}", p0.mu_p);
  LOG("B0 = {}, |B0| = {}", Bp, Bp.length());
  LOG("mp = {}, qm = {}", mp, qm);
  LOG("Eh = {}", Eh);
#endif

  PetscAssertAbort((bool)set_fields, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "DriftKineticPush::set_fields have to be specified");

  Vector3R Vd;
  PetscReal Vh = 0.0;

  PetscReal R1 = 0.0, R2 = 0.0;

  Vector3R B0 = Bp, Bh = Bp;
  Vector3R gradB0 = gradBp, gradBh = gradBp;
  Vector3R b0 = Bp.normalized();
  Vector3R bp = b0, h = b0;

  for (it = 0; it < maxit; ++it) {
    Vh = 0.5 * (pn.p_parallel + p0.p_parallel);
    Vd = get_Vd(p0, h, Vh, Bh.length(), gradBh, Eh);

    R1 = get_residue_r(dt, pn, p0, Vh * h, Vd);
    R2 = get_residue_v(dt, pn, p0, Vh, h, Vd, B0, Eh);

#if 0
    LOG("--- Iteration {} ---", it);
    LOG("R1 = {}, R2 = {}", R1, R2);
#endif

    if ((R1 < eps) && (R2 < delta) && it) {
      return;
    }

    update_r(dt, pn, p0, Vh * h, Vd);

    set_fields(p0.r, pn.r, Eh, Bp, gradBp);
    Bh = 0.5 * (Bp + B0);
    gradBh = 0.5 * (gradBp + gradB0);
    bp = Bp.normalized();
    h = 0.5 * (bp + b0);

    update_v_perp(pn, p0, B0);
    update_v_parallel(dt, pn, p0, Vh, h, Vd, B0, Eh);
  }

  PetscCheckAbort((R1 >= eps) || (R2 >= delta), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "DriftKineticPush::process() nonlinear iterations diverged with norm %e and %e!", R1, R2);
}


Vector3R DriftKineticPush::get_Vd(const PointByField& p0, const Vector3R& h,
  PetscReal Vh, PetscReal Bh, const Vector3R& gradBh, const Vector3R& Eh) const
{
  if (Bh < 1e-12)
    return Vector3R{};

  return Eh.cross(h) / Bh +
    1.0 / qm * (Vh * Vh / Bh + p0.mu_p / mp) * h.cross(gradBh / Bh);
}

void DriftKineticPush::update_r(PetscReal dt, PointByField& pn,
  const PointByField& p0, const Vector3R& Vh, const Vector3R& Vd) const
{
  pn.r = p0.r + dt * (Vh + Vd);
}

void DriftKineticPush::update_v_perp(
  PointByField& pn, const PointByField& p0, const Vector3R& B0) const
{
  pn.p_perp = p0.p_perp * std::sqrt(Bp.length() / B0.length());
}

void DriftKineticPush::update_v_parallel(PetscReal dt, PointByField& pn,
  const PointByField& p0, PetscReal Vh, const Vector3R& h, const Vector3R& Vd,
  const Vector3R& B0, const Vector3R& Eh) const
{
  PetscReal term = (std::abs(Vh) < 1e-12) ? 0.0 : (Eh.dot(Vd) / Vh);
  PetscReal dB = Bp.length() - B0.length();
  PetscReal mu_term = (std::abs(Vh) < 1e-12) ? 0.0 : (p0.mu_p / mp) * (dB / Vh);

  pn.p_parallel = p0.p_parallel + dt * qm * (Eh.dot(h) + term) - mu_term;
}

PetscReal DriftKineticPush::get_residue_r(PetscReal dt, const PointByField& pn,
  const PointByField& p0, const Vector3R& Vh, const Vector3R& Vd) const
{
  return (pn.r - p0.r - dt * (Vh + Vd)).length();
}

PetscReal DriftKineticPush::get_residue_v(PetscReal dt, const PointByField& pn,
  const PointByField& p0, PetscReal Vh, const Vector3R& h, const Vector3R& Vd,
  const Vector3R& B0, const Vector3R& Eh) const
{
  PetscReal term = (std::abs(Vh) < 1e-12) ? 0.0 : (Eh.dot(Vd) / Vh);
  PetscReal dB = Bp.length() - B0.length();
  PetscReal mu_term = (std::abs(Vh) < 1e-12) ? 0.0 : (p0.mu_p / mp) * (dB / Vh);

  return std::abs(
    (pn.p_parallel - p0.p_parallel) - dt * qm * (Eh.dot(h) + term) + mu_term);
}
