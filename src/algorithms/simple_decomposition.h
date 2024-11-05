#ifndef SRC_ALGORITHMS_SIMPLE_DECOMPOSITION_H
#define SRC_ALGORITHMS_SIMPLE_DECOMPOSITION_H

#include "src/utils/particle_shape.h"

/**
 * @brief Simple (dump) density decomposition using shifted and non-shifted shapes.
 * @note The important distinctions from `Esirkepov_decomposition` are that single-
 * particle current is passed as a `J_p` to constructor and `Node::r` is passed
 * into `process()`, because offset here are manually processed.
 */
class Simple_decomposition {
public:
  Simple_decomposition(const Vector3I& width, const Vector3R& J_p,
    const Shape& no, const Shape& sh);

  /// @brief Decomposition context is reference to outer global current.
  using Context = Vector3R***;

  /// @note Here you should pass `p_r` since we would explicitly calculate offsets.
  void process(const Vector3R& p_r, Context& j_field) const;

private:
  const Vector3I& width;

  Vector3R J_p;
  const Shape& no;
  const Shape& sh;
};

#endif  // SRC_ALGORITHMS_SIMPLE_DECOMPOSITION_H
