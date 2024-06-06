#include "region_operations.h"


bool is_point_within_bounds(const Vector3I& point, const Vector3I& b_start, const Vector3I& b_size) {
  Vector3I b_end = b_start + b_size;
  return
    (b_start[X] <= point[X] && point[X] <= b_end[X]) &&
    (b_start[Y] <= point[Y] && point[Y] <= b_end[Y]) &&
    (b_start[Z] <= point[Z] && point[Z] <= b_end[Z]);
}


bool is_region_within_bounds(
    const Vector3I& r_start, const Vector3I& r_size,
    const Vector3I& b_start, const Vector3I& b_size) {
  Vector3I r_end = r_start + r_size;
  Vector3I b_end = b_start + b_size;
  return
    (b_start[X] <= r_start[X] && r_start[X] < b_end[X]) &&
    (b_start[Y] <= r_start[Y] && r_start[Y] < b_end[Y]) &&
    (b_start[Z] <= r_start[Z] && r_start[Z] < b_end[Z]) &&
    (b_start[X] <= r_end[X] && r_end[X] <= b_end[X]) &&
    (b_start[Y] <= r_end[Y] && r_end[Y] <= b_end[Y]) &&
    (b_start[Z] <= r_end[Z] && r_end[Z] <= b_end[Z]);
}


bool is_region_intersect_bounds(
    const Vector3I& r_start, const Vector3I& r_size,
    const Vector3I& b_start, const Vector3I& b_size) {
  Vector3I r_end = r_start + r_size;
  Vector3I b_end = b_start + b_size;
  return
    r_start[X] < b_end[X] && r_end[X] > b_start[X] &&
    r_start[Y] < b_end[Y] && r_end[Y] > b_start[Y] &&
    r_start[Z] < b_end[Z] && r_end[Z] > b_start[Z];
}
