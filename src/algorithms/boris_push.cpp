#include "boris_push.h"

BorisPush::BorisPush(PetscReal dt, const Vector3R& E_p, const Vector3R& B_p)
  : dt(dt), E_p(E_p), B_p(B_p)
{
}

PetscErrorCode BorisPush::process(Point& point, const Context& particles) const
{
  PetscFunctionBeginHot;
  update_u(point, false, particles);
  point.r += point.p * dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode BorisPush::process_rel(Point& point, const Context& particles) const
{
  PetscFunctionBeginHot;
  update_u(point, true, particles);
  point.r += particles.velocity(point) * dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

void BorisPush::update_u(
  Point& point, bool need_gamma, const Context& particles) const
{
  PetscReal m = particles.mass(point);
  PetscReal alpha = 0.5 * dt * particles.charge(point) / m;

  Vector3R& u = point.p;

  Vector3R t = u + alpha * E_p;
  Vector3R b = alpha * B_p;

  if (need_gamma)
    b /= sqrt(1.0 + t.squared() / POW2(m));

  u = 2.0 * (t + t.cross(b) + b * t.dot(b)) / (1.0 + b.squared()) - u;
}

void BorisPush::update_fields(const Vector3R& E_p, const Vector3R& B_p)
{
  this->E_p = E_p;
  this->B_p = B_p;
}

void BorisPush::update_r(
  PetscReal dt, Point& point, const Context& /* particles */)
{
  point.r += point.p * dt;
}

void BorisPush::update_vM(PetscReal dt, Point& point, const Context& particles)
{
  PetscReal theta = get_omega(point, particles) * dt;
  update_v_impl(point.p, std::sin(theta), std::cos(theta));
}

void BorisPush::update_vB(PetscReal dt, Point& point, const Context& particles)
{
  auto [sin, cos] = get_theta_b(dt, point, particles);
  update_v_impl(point.p, sin, cos);
}

void BorisPush::update_vC1(PetscReal dt, Point& point, const Context& particles)
{
  auto [sin, cos] = get_theta_c1(dt, point, particles);
  update_v_impl(point.p, sin, cos);
}

/// @note Separate update is needed because we use different formulas for `theta_c`.
/// This difference arised since we pass `dt/2` into the velocity update, M2B scheme uses the `dt` instead.
void BorisPush::update_vC2(PetscReal dt, Point& point, const Context& particles)
{
  auto [sin, cos] = get_theta_c2(dt, point, particles);
  update_v_impl(point.p, sin, cos);
}

void BorisPush::update_vEB(PetscReal dt, Point& point, const Context& particles)
{
  PetscReal alpha = dt * particles.charge(point) / particles.mass(point);
  Vector3R a = +alpha * E_p;
  Vector3R b = -alpha * B_p;

  Vector3R& v = point.p;
  Vector3R w = v + 0.5 * a;
  v += a + (b.cross(w) + 0.5 * b.cross(b.cross(w))) / (1.0 + 0.25 * b.squared());
}


inline PetscReal BorisPush::get_omega(
  const Point& point, const Context& particles) const
{
  return (-1.0) * particles.charge(point) * B_p.length() / particles.mass(point);
}

std::pair<REP2(PetscReal)> BorisPush::get_theta_b(
  PetscReal dt, const Point& point, const Context& particles) const
{
  PetscReal theta = get_omega(point, particles) * dt;
  PetscReal d = (1.0 + 0.25 * POW2(theta));
  return std::make_pair(theta / d, (1.0 - 0.25 * POW2(theta)) / d);
}

std::pair<REP2(PetscReal)> BorisPush::get_theta_c1(
  PetscReal dt, const Point& point, const Context& particles) const
{
  PetscReal theta = get_omega(point, particles) * dt;
  return std::make_pair(
    theta * std::sqrt(1.0 - 0.25 * POW2(theta)), (1 - 0.5 * POW2(theta)));
}

std::pair<REP2(PetscReal)> BorisPush::get_theta_c2(
  PetscReal dt, const Point& point, const Context& particles) const
{
  PetscReal theta = get_omega(point, particles) * dt;
  return std::make_pair(theta, std::sqrt(1.0 - POW2(theta)));
}

void BorisPush::update_v_impl(
  Vector3R& v, PetscReal sin_theta, PetscReal cos_theta) const
{
  Vector3R b = B_p.normalized();
  Vector3R v_p = v.parallel_to(b);
  Vector3R v_t = v.transverse_to(b);
  v = v_p + cos_theta * v_t + sin_theta * b.cross(v_t);
}
