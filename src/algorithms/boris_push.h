#ifndef SRC_ALGORITHMS_BORIS_PUSH_H
#define SRC_ALGORITHMS_BORIS_PUSH_H

#include "src/interfaces/particles.h"
#include "src/interfaces/point.h"

class BorisPush {
public:
  BorisPush() = delete;
  BorisPush(PetscReal dt, const Vector3R& E_p, const Vector3R& B_p);

  /// @brief Pusher context is the particles storage to get parameters from.
  using Context = interfaces::Particles;

  PetscErrorCode process(Point& point, const Context& particles) const;
  PetscErrorCode process_rel(Point& point, const Context& particles) const;

  /// @note The following is the recreation of the published
  /// results, @see https://doi.org/10.1016/j.jcp.2022.111422
  void update_state(PetscReal dt, const Vector3R& E_p, const Vector3R& B_p);
  void update_r(Point& point, const Context& particles);
  void update_vM(Point& point, const Context& particles);
  void update_vB(Point& point, const Context& particles);
  void update_vC(Point& point, const Context& particles);

  PetscReal get_omega(const Point& point, const Context& particles) const;
  PetscReal get_theta(const Point& point, const Context& particles) const;

private:
  void update_u(
    Point& point, bool need_gamma, const Context& particles) const;

  std::pair<REP2(PetscReal)> get_theta_b(
    const Point& point, const Context& particles) const;

  std::pair<REP2(PetscReal)> get_theta_c(
    const Point& point, const Context& particles) const;

  void update_v_impl(Vector3R& v, PetscReal sin_theta, PetscReal cos_theta) const;

  PetscReal dt;
  Vector3R E_p;
  Vector3R B_p;
};

#endif  // SRC_ALGORITHMS_BORIS_PUSH_H
