#include "simple_decomposition.h"

SimpleDecomposition::SimpleDecomposition(const Shape& shape, const Vector3R& J_p)
  : shape(shape), J_p(J_p)
{
}

PetscErrorCode SimpleDecomposition::process(Context& J) const
{
  PetscFunctionBeginHot;
  // clang-format off: @todo create macro/range-based analogue for this loop
  for (PetscInt z = 0; z < shape.size[Z]; ++z) {
  for (PetscInt y = 0; y < shape.size[Y]; ++y) {
  for (PetscInt x = 0; x < shape.size[X]; ++x) {
    PetscInt g_x = shape.start[X] + x;
    PetscInt g_y = shape.start[Y] + y;
    PetscInt g_z = shape.start[Z] + z;

    PetscInt i = shape.s_p(z, y, x);
    Vector3R J_shape = shape.electric(i);

#pragma omp atomic update
      J[g_z][g_y][g_x][X] += J_p.x() * J_shape.x();

#pragma omp atomic update
      J[g_z][g_y][g_x][Y] += J_p.y() * J_shape.y();

#pragma omp atomic update
      J[g_z][g_y][g_x][Z] += J_p.z() * J_shape.z();
  }}}
  // clang-format on
  PetscFunctionReturn(PETSC_SUCCESS);
}
