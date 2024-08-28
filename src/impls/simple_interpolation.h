#ifndef SRC_IMPLS_SIMPLE_INTERPOLATION_H
#define SRC_IMPLS_SIMPLE_INTERPOLATION_H

#include "src/impls/particle_shape.h"
#include "src/utils/vector3.h"

/**
 * @brief Simple interpolator tightly coupled with `Shape` class.
 * Unifies the calling sequence, when the Yee interpolation is needed.
 */
class Simple_interpolation {
public:
  Simple_interpolation(const Vector3I& width, const Shape& no, const Shape& sh);

  /// @brief Interpolation field context is a pair of point and global fields.
  using Context = std::vector<std::pair<Vector3R&, Vector3R*** const&>>;

  PetscErrorCode process(const Vector3I& p_g, const Context& e_fields, const Context& b_fields) const;

private:
  const Vector3I& width;
  const Shape& no;
  const Shape& sh;
};

#endif // SRC_IMPLS_SIMPLE_INTERPOLATION_H
