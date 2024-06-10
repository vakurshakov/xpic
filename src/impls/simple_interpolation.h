#ifndef SRC_IMPLS_SIMPLE_INTERPOLATION_H
#define SRC_IMPLS_SIMPLE_INTERPOLATION_H

#include "src/impls/particle_shape.h"
#include "src/vectors/vector3.h"

class Simple_interpolation {
public:
  Simple_interpolation(const Vector3I& width, const Shape& no, const Shape& sh);

  struct Context {
    using point_global_fields = std::pair<Vector3R&, Vector3R***>;
    std::vector<point_global_fields> e_fields;
    std::vector<point_global_fields> b_fields;
  };

  PetscErrorCode process(const Vector3I& p_g, Context& context) const;

private:
  const Vector3I& width;
  const Shape& no;
  const Shape& sh;
};

#endif // SRC_IMPLS_SIMPLE_INTERPOLATION_H
