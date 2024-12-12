#include <cassert>
#include <cmath>

#include "src/utils/vector3.h"

void update_counter_clockwise(
  const Vector3R& old_r, const Vector3R& new_r, PetscReal& counter_clockwise)
{
  counter_clockwise += (old_r.y() + new_r.y()) * (old_r.x() - new_r.x());
}

bool equal_tol(PetscReal a, PetscReal b, PetscReal tol)
{
  return std::abs(a - b) < tol;
}

bool equal_tol(const Vector3R& a, const Vector3R& b, PetscReal tol)
{
  return //
    std::abs(a[X] - b[X]) < tol && //
    std::abs(a[Y] - b[Y]) < tol && //
    std::abs(a[Z] - b[Z]) < tol;
}
