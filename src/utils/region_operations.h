#ifndef SRC_UTILS_REGION_OPERATIONS_H
#define SRC_UTILS_REGION_OPERATIONS_H

#include "src/vectors/vector3.h"

bool is_point_within_bounds(
  const Vector3I& point,
  const Vector3I& b_start, const Vector3I& b_size);

bool is_region_within_bounds(
  const Vector3I& r_start, const Vector3I& r_size,
  const Vector3I& b_start, const Vector3I& b_size);

bool is_region_intersect_bounds(
  const Vector3I& r_start, const Vector3I& r_size,
  const Vector3I& b_start, const Vector3I& b_size);

#endif // SRC_UTILS_REGION_OPERATIONS_H
