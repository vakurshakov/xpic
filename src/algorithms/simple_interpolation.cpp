#include "simple_interpolation.h"

SimpleInterpolation::SimpleInterpolation(const Shape& shape)
  : shape(shape)
{
}

PetscErrorCode SimpleInterpolation::process(
  const Context& e_fields, const Context& b_fields) const
{
  PetscFunctionBeginHot;
  for (PetscInt i = 0; i < shape.size.elements_product(); ++i) {
    PetscInt g_x = shape.start[X] + i % shape.size[X];
    PetscInt g_y = shape.start[Y] + (i / shape.size[X]) % shape.size[Y];
    PetscInt g_z = shape.start[Z] + (i / shape.size[X]) / shape.size[Y];

    if (!e_fields.empty()) {
      auto&& E_shape = shape.electric(i);

      for (auto&& [E_p, E_g] : e_fields) {
        E_p.x() += E_g[g_z][g_y][g_x][X] * E_shape[X];
        E_p.y() += E_g[g_z][g_y][g_x][Y] * E_shape[Y];
        E_p.z() += E_g[g_z][g_y][g_x][Z] * E_shape[Z];
      }
    }

    if (!b_fields.empty()) {
      auto&& B_shape = shape.magnetic(i);

      for (auto&& [B_p, B_g] : b_fields) {
        B_p.x() += B_g[g_z][g_y][g_x][X] * B_shape[X];
        B_p.y() += B_g[g_z][g_y][g_x][Y] * B_shape[Y];
        B_p.z() += B_g[g_z][g_y][g_x][Z] * B_shape[Z];
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
