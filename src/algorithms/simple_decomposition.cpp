#include "simple_decomposition.h"

SimpleDecomposition::SimpleDecomposition(const Shape& shape, const Vector3R& J_p)
  : shape(shape), J_p(J_p)
{
}

PetscErrorCode SimpleDecomposition::process(Context& J) const
{
  PetscFunctionBeginHot;
  for (PetscInt i = 0; i < shape.size.elements_product(); ++i) {
    PetscInt g_x = shape.start[X] + i % shape.size[X];
    PetscInt g_y = shape.start[Y] + (i / shape.size[X]) % shape.size[Y];
    PetscInt g_z = shape.start[Z] + (i / shape.size[X]) / shape.size[Y];

    Vector3R J_shape = shape.electric(i);

#pragma omp atomic update
    J[g_z][g_y][g_x][X] += J_p.x() * J_shape.x();

#pragma omp atomic update
    J[g_z][g_y][g_x][Y] += J_p.y() * J_shape.y();

#pragma omp atomic update
    J[g_z][g_y][g_x][Z] += J_p.z() * J_shape.z();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
