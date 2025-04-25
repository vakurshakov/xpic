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

PetscInt CrankNicolsonPush::get_iteration_number() const
{
  return it;
}

void CrankNicolsonPush::set_fields_callback(SetFields&& callback)
{
  this->set_fields = std::move(callback);
}

void CrankNicolsonPush::process(PetscReal dt, Point& pn, const Point& p0)
{
  PetscAssertAbort((bool)set_fields, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "CrankNicolsonPush::set_fields have to be specified");

  PetscReal r0 = 0, rn = 0;

  set_fields(0.5 * (pn.r + p0.r), E_p, B_p);
  r0 = get_residue(dt, pn, p0);

  for (it = 0; it < maxit; ++it) {
    update_v(dt, pn.p, p0.p);
    update_r(dt, pn, p0);
    rn = get_residue(dt, pn, p0);

    if (rn < atol + rtol * r0)
      return;

    set_fields(0.5 * (pn.r + p0.r), E_p, B_p);
  }

  PetscCheckAbort(rn >= atol + rtol * r0, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "CrankNicolsonPush::process() nonlinear iterations diverged with norm %e!", rn);
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
  return ((pn.p - p0.p) / dt - qm * (E_p + vh.cross(B_p))).length();
}
