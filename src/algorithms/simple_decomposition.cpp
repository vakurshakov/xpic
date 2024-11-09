#include "simple_decomposition.h"

Simple_decomposition::Simple_decomposition(
  const Vector3I& width, const Vector3R& p_j, const Shape& no, const Shape& sh)
  : width(width), J_p(p_j), no(no), sh(sh)
{
}

PetscErrorCode Simple_decomposition::process(const Vector3I& p_g, Context& J) const
{
  PetscFunctionBeginHot;
  PetscInt g_x, g_y, g_z;

  // clang-format off: @todo create macro/range-based analogue for this loop
  for (PetscInt z = 0; z < width[Z]; ++z) {
  for (PetscInt y = 0; y < width[Y]; ++y) {
  for (PetscInt x = 0; x < width[X]; ++x) {
    PetscInt i = Shape::index(z, y, x);

    Vector3R J_shape{
      no(i, Z) * no(i, Y) * sh(i, X),
      no(i, Z) * sh(i, Y) * no(i, X),
      sh(i, Z) * no(i, Y) * no(i, X),
    };

    g_x = p_g[X] + x;
    g_y = p_g[Y] + y;
    g_z = p_g[Z] + z;

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
