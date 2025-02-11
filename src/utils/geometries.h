#ifndef SRC_UTILS_GEOMETRIES_H
#define SRC_UTILS_GEOMETRIES_H

#include "src/pch.h"
#include "src/utils/vector3.h"

struct BoxGeometry {
  Vector3R min;
  Vector3R max;
};

struct CircleGeometry {
  Vector3R center;
  PetscReal radius;
};

struct CylinderGeometry {
  Vector3R center;
  PetscReal radius;
  PetscReal height;
};

#endif // SRC_UTILS_GEOMETRIES_H
