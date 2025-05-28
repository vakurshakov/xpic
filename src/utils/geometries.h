#ifndef SRC_UTILS_GEOMETRIES_H
#define SRC_UTILS_GEOMETRIES_H

#include "src/pch.h"
#include "src/utils/vector3.h"

struct BoxGeometry {
  Vector3R min;
  Vector3R max;
};

struct CylinderGeometry {
  Vector3R center;
  PetscReal radius;
  PetscReal height;
};

struct AnnulusGeometry {
  Vector3R center;
  PetscReal inner_r;
  PetscReal outer_r;
  PetscReal height;
};


struct WithinBox {
  bool operator()(const Vector3R& r);
  BoxGeometry geom;
};

struct WithinCylinder {
  bool operator()(const Vector3R& r);
  CylinderGeometry geom;
};

struct WithinAnnulus {
  bool operator()(const Vector3R& r);
  AnnulusGeometry geom;
};


bool is_point_within_bounds(
  const Vector3I& point, const Vector3I& b_start, const Vector3I& b_size);

bool is_region_within_bounds( //
  const Vector3I& r_start, const Vector3I& r_size, //
  const Vector3I& b_start, const Vector3I& b_size);

bool is_region_intersect_bounds( //
  const Vector3I& r_start, const Vector3I& r_size, //
  const Vector3I& b_start, const Vector3I& b_size);

#endif // SRC_UTILS_GEOMETRIES_H
