#include "point.h"

Point::Point(const Vector3R& r, const Vector3R& p)
  : r(r), p(p)
{
}

void g_bound_reflective(Point& point, Axis axis)
{
  PetscReal& s = point.r[axis];
  PetscReal& ps = point.p[axis];

  if (s < 0.0) {
    s = 0.0;
    ps *= -1.0;
  }
  else if (s > Geom[axis]) {
    s = Geom[axis];
    ps *= -1.0;
  }
}

void g_bound_periodic(Point& point, Axis axis)
{
  PetscReal& s = point.r[axis];

  if (s < 0.0)
    s = Geom[axis] - (0.0 - s);
  else if (s > Geom[axis])
    s = 0.0 + (s - Geom[axis]);
}
