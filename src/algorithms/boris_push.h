#ifndef SRC_ALGORITHMS_BORIS_PUSH_H
#define SRC_ALGORITHMS_BORIS_PUSH_H

#include "src/interfaces/particles.h"
#include "src/interfaces/point.h"

class BorisPush {
public:
  BorisPush() = default;
  BorisPush(PetscReal dt, const Vector3R& E_p, const Vector3R& B_p);

  /// @brief Pusher context is the particles storage to get parameters from.
  using Context = interfaces::Particles;

  PetscErrorCode process(Point& point, const Context& particles) const;
  PetscErrorCode process_rel(Point& point, const Context& particles) const;

  /// @note The following is the recreation of the published
  /// results, @see https://doi.org/10.1016/j.jcp.2022.111422
  void update_fields(const Vector3R& E_p, const Vector3R& B_p);
  void update_r(PetscReal dt, Point& point, const Context& particles);

  /// @note Magnetic field integrators
  void update_vM(PetscReal dt, Point& point, const Context& particles);
  void update_vB(PetscReal dt, Point& point, const Context& particles);
  void update_vC1(PetscReal dt, Point& point, const Context& particles);
  void update_vC2(PetscReal dt, Point& point, const Context& particles);

  /// @note Electro-magnetic field integrator. It substitutes
  /// Boris angle `theta_b` into the velocity update _directly_.
  void update_vEB(PetscReal dt, Point& point, const Context& particles);

private:
  void update_u(Point& point, bool need_gamma, const Context& particles) const;

  PetscReal get_omega(const Point& point, const Context& particles) const;

  std::pair<REP2(PetscReal)> get_theta_b(
    PetscReal dt, const Point& point, const Context& particles) const;

  std::pair<REP2(PetscReal)> get_theta_c1(
    PetscReal dt, const Point& point, const Context& particles) const;

  std::pair<REP2(PetscReal)> get_theta_c2(
    PetscReal dt, const Point& point, const Context& particles) const;

  void update_v_impl(Vector3R& v, PetscReal sin_theta, PetscReal cos_theta) const;

  PetscReal dt;
  Vector3R E_p;
  Vector3R B_p;
};

#endif  // SRC_ALGORITHMS_BORIS_PUSH_H
