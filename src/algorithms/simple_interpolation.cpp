#include "simple_interpolation.h"

Simple_interpolation::Simple_interpolation(const Shape& shape)
  : shape(shape)
{
}

PetscErrorCode Simple_interpolation::process(
  const Context& e_fields, const Context& b_fields) const
{
  PetscFunctionBeginHot;
  // clang-format off: @todo create macro/range-based analogue for this loop
  for (PetscInt z = 0; z < shape.size[Z]; ++z) {
  for (PetscInt y = 0; y < shape.size[Y]; ++y) {
  for (PetscInt x = 0; x < shape.size[X]; ++x) {
    PetscInt i = shape.s_p(z, y, x);
    PetscInt g_x = shape.start[X] + x;
    PetscInt g_y = shape.start[Y] + y;
    PetscInt g_z = shape.start[Z] + z;

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
  }}}
  // clang-format on
  PetscFunctionReturn(PETSC_SUCCESS);
}
