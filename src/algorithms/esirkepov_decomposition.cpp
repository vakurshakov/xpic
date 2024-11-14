#include "esirkepov_decomposition.h"

Esirkepov_decomposition::Esirkepov_decomposition(PetscInt width,
  PetscReal alpha, const Shape& old_shape, const Shape& new_shape)
  : width(width), alpha(alpha), old_shape(old_shape), new_shape(new_shape)
{
}

PetscErrorCode Esirkepov_decomposition::process(
  const Vector3I& p_g, Context& J) const
{
  PetscFunctionBeginHot;
  static constexpr PetscInt j_width = POW2(shape_width);
  static PetscReal temp_j[j_width * Vector3R::dim];
#pragma omp threadprivate(temp_j)

  PetscInt g_x, g_y, g_z;

  // clang-format off: @todo create macro/range-based analogue for this loop
  for (PetscInt z = 0; z < width; ++z) {
  for (PetscInt y = 0; y < width; ++y) {
  for (PetscInt x = 0; x < width; ++x) {
    g_x = p_g[X] + x;
    g_y = p_g[Y] + y;
    g_z = p_g[Z] + z;

    PetscReal p_jx = get_Jx(x, y, z, temp_j + j_width * X);
    PetscReal p_jy = get_Jy(x, y, z, temp_j + j_width * Y);
    PetscReal p_jz = get_Jz(x, y, z, temp_j + j_width * Z);

#pragma omp atomic update
      J[g_z][g_y][g_x][X] += p_jx;

#pragma omp atomic update
      J[g_z][g_y][g_x][Y] += p_jy;

#pragma omp atomic update
      J[g_z][g_y][g_x][Z] += p_jz;
  }}}
  // clang-format on
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscReal Esirkepov_decomposition::get_Jx(
  PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jx) const
{
  PetscFunctionBeginHot;
  PetscReal qx = alpha * dx;
  PetscInt i = ((z * shape_width + y) * shape_width + x);
  PetscInt j = (z * shape_width + y);

  PetscReal wx_p = -qx * (new_shape(i, X) - old_shape(i, X)) *
    (new_shape(i, Y) * (2.0 * new_shape(i, Z) + old_shape(i, Z)) +
      old_shape(i, Y) * (2.0 * old_shape(i, Z) + new_shape(i, Z)));

  temp_jx[j] = ((x > 0) * temp_jx[j]) + wx_p;
  PetscFunctionReturn(temp_jx[i]);
}

PetscReal Esirkepov_decomposition::get_Jy(
  PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jy) const
{
  PetscFunctionBeginHot;
  PetscReal qy = alpha * dy;
  PetscInt i = ((z * shape_width + y) * shape_width + x);
  PetscInt j = (z * shape_width + x);

  PetscReal wy_p = -qy * (new_shape(i, Y) - old_shape(i, Y)) *
    (new_shape(i, X) * (2.0 * new_shape(i, Z) + old_shape(i, Z)) +
      old_shape(i, X) * (2.0 * old_shape(i, Z) + new_shape(i, Z)));

  temp_jy[j] = ((y > 0) * temp_jy[j]) + wy_p;
  PetscFunctionReturn(temp_jy[j]);
}

PetscReal Esirkepov_decomposition::get_Jz(
  PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jz) const
{
  PetscFunctionBeginHot;
  PetscReal qz = alpha * dz;
  PetscInt i = ((z * shape_width + y) * shape_width + x);
  PetscInt j = (y * shape_width + x);

  PetscReal wz_p = -qz * (new_shape(i, Z) - old_shape(i, Z)) *
    (new_shape(i, Y) * (2.0 * new_shape(i, X) + old_shape(i, X)) +
      old_shape(i, Y) * (2.0 * old_shape(i, X) + new_shape(i, X)));

  temp_jz[j] = ((z > 0) * temp_jz[j]) + wz_p;
  PetscFunctionReturn(temp_jz[j]);
}
