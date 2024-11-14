#include "simple_interpolation.h"

Simple_interpolation::Simple_interpolation(
  PetscInt width, const Shape& no, const Shape& sh)
  : width(width), no(no), sh(sh)
{
}

PetscErrorCode Simple_interpolation::process(
  const Vector3I& p_g, const Context& e_fields, const Context& b_fields) const
{
  PetscFunctionBeginHot;
  PetscInt g_x, g_y, g_z;

  // clang-format off: @todo create macro/range-based analogue for this loop
  for (PetscInt z = 0; z < width; ++z) {
  for (PetscInt y = 0; y < width; ++y) {
  for (PetscInt x = 0; x < width; ++x) {
    PetscInt i = indexing::s_p(z, y, x, width);
    g_x = p_g[X] + x;
    g_y = p_g[Y] + y;
    g_z = p_g[Z] + z;

    if (!e_fields.empty()) {
      Vector3R E_shape{
        no(i, Z) * no(i, Y) * sh(i, X),
        no(i, Z) * sh(i, Y) * no(i, X),
        sh(i, Z) * no(i, Y) * no(i, X),
      };

      for (auto&& [E_p, E_g] : e_fields) {
        E_p.x() += E_g[g_z][g_y][g_x][X] * E_shape.x();
        E_p.y() += E_g[g_z][g_y][g_x][Y] * E_shape.y();
        E_p.z() += E_g[g_z][g_y][g_x][Z] * E_shape.z();
      }
    }

    if (!b_fields.empty()) {
      Vector3R B_shape{
        sh(i, Z) * sh(i, Y) * no(i, X),
        sh(i, Z) * no(i, Y) * sh(i, X),
        no(i, Z) * sh(i, Y) * sh(i, X),
      };

      for (auto&& [B_p, B_g] : b_fields) {
        B_p.x() += B_g[g_z][g_y][g_x][X] * B_shape.x();
        B_p.y() += B_g[g_z][g_y][g_x][Y] * B_shape.y();
        B_p.z() += B_g[g_z][g_y][g_x][Z] * B_shape.z();
      }
    }
  }}}
  // clang-format on
  PetscFunctionReturn(PETSC_SUCCESS);
}
