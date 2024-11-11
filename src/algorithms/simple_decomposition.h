#ifndef SRC_ALGORITHMS_SIMPLE_DECOMPOSITION_H
#define SRC_ALGORITHMS_SIMPLE_DECOMPOSITION_H

#include "src/utils/particle_shape.h"

/// @brief Simple (dump) density decomposition using shifted and non-shifted shapes.
class Simple_decomposition {
public:
  Simple_decomposition(const Vector3I& width, const Vector3R& J_p,
    const Shape& no, const Shape& sh);

  /// @brief Decomposition context is a reference to outer global current.
  using Context = Vector3R***;

  PetscErrorCode process(const Vector3I& p_g, Context& J) const;

private:
  const Vector3I& width;

  Vector3R J_p;
  const Shape& no;
  const Shape& sh;
};

#endif  // SRC_ALGORITHMS_SIMPLE_DECOMPOSITION_H
