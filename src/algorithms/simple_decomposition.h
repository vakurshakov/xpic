#ifndef SRC_ALGORITHMS_SIMPLE_DECOMPOSITION_H
#define SRC_ALGORITHMS_SIMPLE_DECOMPOSITION_H

#include "src/utils/particle_shape.h"

/**
 * @brief Simple (dump) density decomposition using shifted and non-shifted shapes.
 * There is no additional memory-cost of temporary storage like in `Esirkepov_decomposition`.
 */
class Simple_decomposition {
public:
  Simple_decomposition(const Shape& shape, const Vector3R& J_p);

  /// @brief Decomposition context is a reference to outer global current.
  using Context = Vector3R***;

  PetscErrorCode process(Context& J) const;

private:
  const Shape& shape;
  Vector3R J_p;
};

#endif  // SRC_ALGORITHMS_SIMPLE_DECOMPOSITION_H
