#ifndef SRC_UTILS_GEOMETRIES_H
#define SRC_UTILS_GEOMETRIES_H

#include "src/pch.h"
#include "src/utils/vector3.h"

struct CircleGeometry {
  CircleGeometry(const Vector3R& center, PetscReal radius);

  Vector3R center;
  PetscReal radius;
};

#endif // SRC_UTILS_GEOMETRIES_H