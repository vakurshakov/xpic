#ifndef SRC_ALGORITHMS_BORIS_PUSH_H
#define SRC_ALGORITHMS_BORIS_PUSH_H

#include "src/interfaces/particles.h"
#include "src/interfaces/point.h"

/// @note For a detailed classification of boris schemes we refer to
/// https://doi.org/10.1016/j.jcp.2022.111422 and tests/boris_push/boris_push.h:201
class BorisPush {
public:
  BorisPush() = default;
  BorisPush(PetscReal qm, const Vector3R& E_p, const Vector3R& B_p);

  /// @brief Sets charge to mass ratio, it is constant during the push.
  /// @note This implies non-relativistic limit, where `q/m` defines the motion.
  void set_qm(PetscReal qm);

  /// @brief Sets particle-local fields, they are constant in the push.
  void set_fields(const Vector3R& E_p, const Vector3R& B_p);

  static void update_r(PetscReal dt, Point& point);

  /// @brief Magnetic field integrators.
  void update_vM(PetscReal dt, Point& point) const;
  void update_vB(PetscReal dt, Point& point) const;
  void update_vC1(PetscReal dt, Point& point) const;
  void update_vC2(PetscReal dt, Point& point) const;

  /// @brief Electro-magnetic field integrator.
  void update_vEB(PetscReal dt, Point& point) const;

  /// @brief Relativistic move that updates _both_ momentum and coordinate.
  void process(PetscReal dt, Point& point,  //
    const interfaces::Particles& particles) const;

private:
  using AnglePair = std::pair<PetscReal, PetscReal>;
  PetscReal get_theta(PetscReal dt) const;
  AnglePair get_theta_b(PetscReal dt) const;
  AnglePair get_theta_c1(PetscReal dt) const;
  AnglePair get_theta_c2(PetscReal dt) const;
  void update_v_impl(Vector3R& v, const AnglePair& pair) const;

  PetscReal qm = 0;
  Vector3R E_p;
  Vector3R B_p;
};

#endif  // SRC_ALGORITHMS_BORIS_PUSH_H
