#ifndef SRC_ALGORITHMS_SIMPLE_INTERPOLATION_H
#define SRC_ALGORITHMS_SIMPLE_INTERPOLATION_H

#include "src/utils/shape.h"

/**
 * @brief Simple interpolator tightly coupled with `Shape` class.
 * Unifies the calling sequence, when the Yee interpolation is needed.
 */
class Simple_interpolation {
public:
  Simple_interpolation(const Shape& shape);

  /// @brief Interpolation field context is a pair of point and global fields.
  using Context = std::vector<std::pair<Vector3R&, Vector3R*** const&>>;

  PetscErrorCode process(const Context& e_fields, const Context& b_fields) const;

private:
  const Shape& shape;
};

#endif  // SRC_ALGORITHMS_SIMPLE_INTERPOLATION_H
