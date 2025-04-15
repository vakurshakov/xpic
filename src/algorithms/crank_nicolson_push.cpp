#include "crank_nicolson_push.h"

CrankNicolsonPush::CrankNicolsonPush(PetscReal qm)
  : qm(qm)
{
}

void CrankNicolsonPush::set_tolerances(
  PetscReal atol, PetscReal rtol, PetscInt maxit)
{
  this->atol = atol;
  this->rtol = rtol;
  this->maxit = maxit;
}

void CrankNicolsonPush::set_qm(PetscReal qm)
{
  this->qm = qm;
}

void CrankNicolsonPush::set_fields_callback(SetFields&& callback)
{
  this->set_fields = std::move(callback);
}

void CrankNicolsonPush::process(PetscReal dt, Point& pn, const Point& p0)
{
  PetscAssertAbort((bool)set_fields, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "CrankNicolsonPush::set_fields have to be specified");

  set_fields(pn.r, E_p, B_p);
  PetscReal r0 = get_residue(dt, pn, p0);

  for (PetscInt k = 0; k < maxit; ++k) {
    update_v(dt, pn.p, p0.p);
    update_r(dt, pn, p0);

    set_fields(0.5 * (pn.r + p0.r), E_p, B_p);
    PetscReal rn = get_residue(dt, pn, p0);

    if (rn < atol + rtol * r0)
      return;
  }
}

void CrankNicolsonPush::update_v(
  PetscReal dt, Vector3R& vn, const Vector3R& v0) const
{
  PetscReal alpha = 0.5 * dt * qm;
  Vector3R a = alpha * E_p;
  Vector3R b = alpha * B_p;

  Vector3R w = v0 + a;
  Vector3R vh = (w + w.cross(b) + b * w.dot(b)) / (1.0 + b.squared());

  vn = 2.0 * vh - v0;
}

void CrankNicolsonPush::update_r(PetscReal dt, Point& pn, const Point& p0) const
{
  pn.r = p0.r + (0.5 * dt) * (pn.p + p0.p);
}

PetscReal CrankNicolsonPush::get_residue(
  PetscReal dt, const Point& pn, const Point& p0) const
{
  Vector3R vh = 0.5 * (pn.p + p0.p);
  Vector3R rv = ((pn.p - p0.p) / dt - qm * (E_p + vh.cross(B_p)));
  Vector3R rr = ((pn.r - p0.r) / dt - vh);
  return std::hypot(rv.length(), rr.length());
}
