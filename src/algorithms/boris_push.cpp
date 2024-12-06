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


void BorisPush::process_M1A(Point& point, const Context& particles)
{
  PetscReal theta = get_theta(point, particles);
  point.p = get_vb(point.p, std::cos(theta), std::sin(theta));
  point.r += point.p * dt;
}

void BorisPush::process_M1B(Point& point, const Context& particles)
{
  PetscReal theta = get_theta(point, particles);
  point.r += point.p * dt;
  point.p = get_vb(point.p, std::cos(theta), std::sin(theta));
}


inline PetscReal BorisPush::get_omega(
  const Point& point, const Context& particles) const
{
  return (-1.0) * particles.charge(point) * B_p.length() / particles.mass(point);
}

inline PetscReal BorisPush::get_theta(
  const Point& point, const Context& particles) const
{
  return get_omega(point, particles) * dt;
}

inline Vector3R BorisPush::get_vb(
  const Vector3R& v, PetscReal cos_theta, PetscReal sin_theta) const
{
  Vector3R b = B_p.normalized();
  Vector3R v_p = v.parallel_to(b);
  Vector3R v_t = v.transverse_to(b);
  return v_p + cos_theta * v_t + sin_theta * b.cross(v_t);
}
