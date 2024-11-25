#ifndef SRC_ALGORITHMS_BORIS_PUSH_H
#define SRC_ALGORITHMS_BORIS_PUSH_H

#include "src/interfaces/particles.h"
#include "src/interfaces/point.h"

class Boris_push {
public:
  Boris_push() = delete;
  Boris_push(PetscReal dt, const Vector3R& E_p, const Vector3R& B_p);

  /// @brief Pusher context is the particles storage to get parameters from.
  using Context = interfaces::Particles;

  PetscErrorCode process(Point& point, const Context& particles) const;
  PetscErrorCode process_rel(Point& point, const Context& particles) const;

private:
  inline void update_u(
    Point& point, bool need_gamma, const Context& particles) const;

  PetscReal dt;
  const Vector3R& E_p;
  const Vector3R& B_p;
};

#endif  // SRC_ALGORITHMS_BORIS_PUSH_H
