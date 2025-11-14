#include "boris_push.h"

BorisPush::BorisPush(PetscReal qm, const Vector3R& E_p, const Vector3R& B_p)
  : qm(qm), E_p(E_p), B_p(B_p)
{
}

void BorisPush::set_qm(PetscReal qm)
{
  this->qm = qm;
}

void BorisPush::set_fields(const Vector3R& E_p, const Vector3R& B_p)
{
  this->E_p = E_p;
  this->B_p = B_p;
}

/* static */ void BorisPush::update_r(PetscReal dt, Point& point)
{
  point.r += point.p * dt;
}

void BorisPush::update_vM(PetscReal dt, Point& point) const
{
  PetscReal theta = get_theta(dt);
  update_v_impl(point.p, std::make_pair(std::sin(theta), std::cos(theta)));
}

void BorisPush::update_vB(PetscReal dt, Point& point) const
{
  update_v_impl(point.p, get_theta_b(dt));
}

void BorisPush::update_vC1(PetscReal dt, Point& point) const
{
  update_v_impl(point.p, get_theta_c1(dt));
}

/// @details Separate update is needed because we use different
/// formulas for `theta_c`. This difference arose since we pass
/// `dt/2` into the velocity update, M2B scheme uses the `dt` instead.
void BorisPush::update_vC2(PetscReal dt, Point& point) const
{
  update_v_impl(point.p, get_theta_c2(dt));
}

void BorisPush::update_vEB(PetscReal dt, Point& point) const
{
  PetscReal alpha = dt * qm;
  Vector3R a = +alpha * E_p;
  Vector3R b = -alpha * B_p;

  Vector3R& v = point.p;
  Vector3R w = v + 0.5 * a;
  v += a + (b.cross(w) + 0.5 * b.cross(b.cross(w))) / (1.0 + 0.25 * b.squared());
}


inline PetscReal BorisPush::get_theta(PetscReal dt) const
{
  return (-1.0) * qm * B_p.length() * dt;
}

BorisPush::AnglePair BorisPush::get_theta_b(PetscReal dt) const
{
  PetscReal theta = get_theta(dt);
  PetscReal d = (1.0 + 0.25 * POW2(theta));
  return std::make_pair(theta / d, (1.0 - 0.25 * POW2(theta)) / d);
}

BorisPush::AnglePair BorisPush::get_theta_c1(PetscReal dt) const
{
  PetscReal theta = get_theta(dt);
  return std::make_pair(
    theta * std::sqrt(1.0 - 0.25 * POW2(theta)), 1 - 0.5 * POW2(theta));
}

BorisPush::AnglePair BorisPush::get_theta_c2(PetscReal dt) const
{
  PetscReal theta = get_theta(dt);
  return std::make_pair(theta, std::sqrt(1.0 - POW2(theta)));
}

void BorisPush::update_v_impl(Vector3R& v, const BorisPush::AnglePair& pair) const
{
  Vector3R b = B_p.normalized();
  Vector3R v_p = v.parallel_to(b);
  Vector3R v_t = v.transverse_to(b);
  v = v_p + pair.second * v_t + pair.first * b.cross(v_t);
}
