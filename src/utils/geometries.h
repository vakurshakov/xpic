#ifndef SRC_UTILS_GEOMETRIES_H
#define SRC_UTILS_GEOMETRIES_H

#include "src/pch.h"
#include "src/utils/vector3.h"

struct DotGeometry {
  DotGeometry(const Vector3R& dot);
  Vector3R dot;
};

struct BoxGeometry {
  BoxGeometry(const Vector3R& min, const Vector3R& max);
  Vector3R min;
  Vector3R max;
};

struct CircleGeometry {
  CircleGeometry(const Vector3R& center, PetscReal radius);

  Vector3R center;
  PetscReal radius;
};

#endif // SRC_UTILS_GEOMETRIES_H
