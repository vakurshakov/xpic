#ifndef SRC_RICKETSON_PUSH_H
#define SRC_RICKETSON_PUSH_H

/// @todo To be restored and merged with standard Crank-Nicolson mover
#if 0

#include "src/pch.h"
#include "src/interfaces/point.h"
#include "src/utils/shape.h"

class RicketsonPush {
public:
  RicketsonPush() = default;

  void push(Point& pn);

private:
  /**
   * @brief Internal constants controlling the restrictions of the scheme.
   * @details
   * α ∈ (0, 1), controls how close to the time-step restrictions one is willing to get.
   * β ∈ (0, 1), controls the region |v_E / u - 1| < β, where conserving effective force becomes discontinuous.
   * ε ∈ (0, 1), controls the maximum permissible fractional change in μ within a time-step.
   * Γ > 0, measures the accuracy with which we wish to resolve spatial variations in the magnetic field.
   * t_res -- The smallest timescale in the problem that we wish to resolve.
   */
  static constexpr PetscReal alpha = 0.9;
  static constexpr PetscReal beta = 0.2;
  static constexpr PetscReal eps = 0.15;
  static constexpr PetscReal gamma = 0.1;
  static constexpr PetscReal t_res = 10;

  PetscReal m, q;

  const Shape& shape;

  Vector3R x_n, v_n, x_h, v_h;
  Vector3R v_hp, v_ht, v_E, DB_pp, DB_pt;
  PetscReal v_En, u;

  void update(const Vector3R& x_nn, const Vector3R& v_nn);

  void push();
  void adaptive_time_stepping(const Point& point);

  void form_picard_iteration();
};

#endif

#endif  // SRC_RICKETSON_PUSH_H
