#include <cassert>

#include "src/utils/vector3.h"

void update_counter_clockwise(
  const Vector3R& old_r, const Vector3R& new_r, PetscReal& counter_clockwise)
{
  counter_clockwise += (old_r.y() + new_r.y()) * (old_r.x() - new_r.x());
}
