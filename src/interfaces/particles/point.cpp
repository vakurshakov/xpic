#include "point.h"

Point::Point(
  const Vector3R& r,
  const Vector3R& p
#if PARTICLES_LOCAL_PNUM
  , PetscInt Np
#endif
#if PARTICLES_LOCAL_DENSITY
  , PetscReal n
#endif
  )
  : r(r)
  , p(p)
#if PARTICLES_LOCAL_PNUM
  , __Np(Np)
#endif
#if PARTICLES_LOCAL_DENSITY
  , __n(n)
#endif
{
}

void g_bound_reflective(Point& point, Axis axis) {
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

void g_bound_periodic(Point& point, Axis axis) {
  PetscReal& s = point.r[axis];

  if (s < 0.0) {
    s = Geom[axis] - (0.0 - s);
  }
  else if (s > Geom[axis]) {
    s = 0.0 + (s - Geom[axis]);
  }
}
