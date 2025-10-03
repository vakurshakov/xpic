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

  PetscReal rn = 0, r0 = 0;

  Vector3R vh = 0.5 * (pn.p + p0.p);

  auto calc_residue = [&] {
    return ((pn.p - p0.p) / dt - qm * (E_p + vh.cross(B_p))).length();
  };

  set_fields(pn.r, p0.r, E_p, B_p);
  r0 = calc_residue();

  PetscReal alpha = 0.5 * dt * qm;
  Vector3R a, b, w;

  for (it = 0; it < maxit; ++it) {
    a = alpha * E_p;
    b = alpha * B_p;

    w = p0.p + a;
    vh = (w + w.cross(b) + b * w.dot(b)) / (1.0 + b.squared());

    pn.r = p0.r + dt * vh;
    pn.p = 2.0 * vh - p0.p;

    rn = calc_residue();

    if (rn < atol + rtol * r0)
      return;

    set_fields(pn.r, p0.r, E_p, B_p);
  }

  PetscCheckAbort(rn >= atol + rtol * r0, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "CrankNicolsonPush::process() nonlinear iterations diverged with norm %e!", rn);
}
