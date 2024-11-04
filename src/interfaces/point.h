#ifndef SRC_INTERFACES_POINT_H
#define SRC_INTERFACES_POINT_H

#include "src/pch.h"

class Point {
public:
  Vector3R r = 0.0;
  Vector3R p = 0.0;

  Point() = default;
  Point(const Vector3R& r, const Vector3R& p);

  // clang-format off: access modifiers
  PetscReal& x() { return r.x(); }
  PetscReal& y() { return r.y(); }
  PetscReal& z() { return r.z(); }

  PetscReal x() const { return r.x(); }
  PetscReal y() const { return r.y(); }
  PetscReal z() const { return r.z(); }

  PetscReal& px() { return p.x(); }
  PetscReal& py() { return p.y(); }
  PetscReal& pz() { return p.z(); }

  PetscReal px() const { return p.x(); }
  PetscReal py() const { return p.y(); }
  PetscReal pz() const { return p.z(); }
  // clang-format on
};

void g_bound_reflective(Point& point, Axis axis);
void g_bound_periodic(Point& point, Axis axis);

#endif  // SRC_INTERFACES_POINT_H
