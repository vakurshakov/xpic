#include "geometries.h"

DotGeometry::DotGeometry(const Vector3R& dot)
  : dot(dot)
{
}

BoxGeometry::BoxGeometry(const Vector3R& min, const Vector3R& max)
  : min(min), max(max)
{
}

CircleGeometry::CircleGeometry(const Vector3R& center, PetscReal radius)
  : center(center), radius(radius)
{
}

