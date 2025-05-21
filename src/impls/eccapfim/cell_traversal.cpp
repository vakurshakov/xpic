#include "cell_traversal.h"

#include <iostream>
#include <vector>

std::vector<Vector3R> cell_traversal(const Vector3R& end, const Vector3R& start)
{
  Vector3I curr{
    (PetscInt)std::round(start[X] / dx),
    (PetscInt)std::round(start[Y] / dy),
    (PetscInt)std::round(start[Z] / dz),
  };

  Vector3I last{
    (PetscInt)std::round(end[X] / dx),
    (PetscInt)std::round(end[Y] / dy),
    (PetscInt)std::round(end[Z] / dz),
  };

  if (curr == last) {
    return {start, end};
  }

  Vector3R dir = (end - start);
  PetscInt sx = dir[X] > 0 ? 1 : -1;
  PetscInt sy = dir[Y] > 0 ? 1 : -1;
  PetscInt sz = dir[Z] > 0 ? 1 : -1;

  Vector3R next{
    (curr[X] + sx * 0.5) * dx,
    (curr[Y] + sy * 0.5) * dy,
    (curr[Z] + sz * 0.5) * dz,
  };

  static const PetscReal max = std::numeric_limits<double>::max();

  PetscReal t;
  PetscReal tx = (dir[X] != 0) ? (next[X] - start[X]) / dir[X] : max;
  PetscReal ty = (dir[Y] != 0) ? (next[Y] - start[Y]) / dir[Y] : max;
  PetscReal tz = (dir[Z] != 0) ? (next[Z] - start[Z]) / dir[Z] : max;

  PetscReal dtx = (dir[X] != 0) ? dx / dir[X] * sx : 0.0;
  PetscReal dty = (dir[Y] != 0) ? dy / dir[Y] * sy : 0.0;
  PetscReal dtz = (dir[Z] != 0) ? dz / dir[Z] * sz : 0.0;

  std::vector<Vector3R> points;
  points.push_back(start);

  while (curr != last) {
    if (tx < ty) {
      if (tx < tz) {
        t = tx;
        curr[X] += sx;
        tx += dtx;
      }
      else {
        t = tz;
        curr[Z] += sz;
        tz += dtz;
      }
    }
    else {
      if (ty < tz) {
        t = ty;
        curr[Y] += sy;
        ty += dty;
      }
      else {
        t = tz;
        curr[Z] += sz;
        tz += dtz;
      }
    }

    points.push_back(start + dir * t);
  }

  points.push_back(end);
  return points;
}
