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
  void process_M1A(Point& point, const Context& particles);
  void process_M1B(Point& point, const Context& particles);
  void process_MLF(Point& point, const Context& particles);
  void process_B1A(Point& point, const Context& particles);
  void process_B1B(Point& point, const Context& particles);
  void process_BLF(Point& point, const Context& particles);
  void process_C1A(Point& point, const Context& particles);
  void process_C1B(Point& point, const Context& particles);
  void process_CLF(Point& point, const Context& particles);

  PetscReal get_omega(const Point& point, const Context& particles) const;
  PetscReal get_theta(const Point& point, const Context& particles) const;

private:
  inline void update_u(
    Point& point, bool need_gamma, const Context& particles) const;

  void impl_M1A(Point& point, PetscReal sin_theta, PetscReal cos_theta);
  void impl_M1B(Point& point, PetscReal sin_theta, PetscReal cos_theta);

  std::pair<REP2(PetscReal)> get_theta_b(
    const Point& point, const Context& particles) const;

  std::pair<REP2(PetscReal)> get_theta_c(
    const Point& point, const Context& particles) const;

  Vector3R get_vb(
    const Vector3R& v, PetscReal sin_theta, PetscReal cos_theta) const;

  PetscReal dt;
  const Vector3R& E_p;
  const Vector3R& B_p;
};

#endif  // SRC_ALGORITHMS_BORIS_PUSH_H
