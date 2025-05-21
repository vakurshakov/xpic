#ifndef SRC_ALGORITHMS_IMPLICIT_ESIRKEPOV_H
#define SRC_ALGORITHMS_IMPLICIT_ESIRKEPOV_H

#include "src/interfaces/point.h"
#include "src/utils/shape.h"

namespace ImplicitEsirkepov {

PetscErrorCode interpolation(Vector3R& E_p, Vector3R*** E_g, //
  const Vector3R& rn, const Vector3R& r0);

PetscErrorCode decomposition(Vector3R*** J_g, //
  const Vector3R& rn, const Vector3R& r0, const Vector3R& v, PetscReal alpha);

}  // namespace ImplicitEsirkepov

#endif  // SRC_ALGORITHMS_IMPLICIT_ESIRKEPOV_H

