#include "esirkepov_decomposition.h"

namespace indexing {

constexpr PetscInt j_p(PetscInt x1, PetscInt x2)
{
  return x2 * shape_width + x1;
}

}  // namespace indexing


EsirkepovDecomposition::EsirkepovDecomposition(const Shape& shape, PetscReal alpha)
  : shape(shape), alpha(alpha)
{
}

/// @todo this can be replaced with the same `VecSetValuesCOO()` approach
/// @todo there are lots of empty p_j values, they should condensed in some way
PetscErrorCode EsirkepovDecomposition::process(Context& J) const
{
  PetscFunctionBeginHot;
  static constexpr PetscInt j_geom = POW2(shape_width);
  static constexpr PetscInt j_size = Vector3I::dim * j_geom;

  PetscReal temp_j[j_size];
  std::fill_n(temp_j, j_size, 0.0);

  for (PetscInt i = 0; i < shape.size.elements_product(); ++i) {
    PetscInt x = i % shape.size[X];
    PetscInt y = (i / shape.size[X]) % shape.size[Y];
    PetscInt z = (i / shape.size[X]) / shape.size[Y];

    PetscInt g_x = shape.start[X] + x;
    PetscInt g_y = shape.start[Y] + y;
    PetscInt g_z = shape.start[Z] + z;

    Vector3R j_p{
      get_jx(x, y, z, temp_j + j_geom * X),
      get_jy(x, y, z, temp_j + j_geom * Y),
      get_jz(x, y, z, temp_j + j_geom * Z),
    };

#pragma omp atomic update
    J[g_z][g_y][g_x][X] += j_p[X];

#pragma omp atomic update
    J[g_z][g_y][g_x][Y] += j_p[Y];

#pragma omp atomic update
    J[g_z][g_y][g_x][Z] += j_p[Z];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscReal EsirkepovDecomposition::get_jx(
  PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jx) const
{
  PetscFunctionBeginHot;
  PetscReal qx = alpha * dx;
  PetscInt i = shape.s_p(x, y, z);
  PetscInt j = indexing::j_p(y, z);

  PetscReal wx_p = -qx * (shape(i, New, X) - shape(i, Old, X)) *
    (shape(i, New, Y) * (2.0 * shape(i, New, Z) + shape(i, Old, Z)) +
      shape(i, Old, Y) * (2.0 * shape(i, Old, Z) + shape(i, New, Z)));

  temp_jx[j] = (static_cast<PetscReal>(x > 0) * temp_jx[j]) + wx_p;
  PetscFunctionReturn(temp_jx[j]);
}

PetscReal EsirkepovDecomposition::get_jy(
  PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jy) const
{
  PetscFunctionBeginHot;
  PetscReal qy = alpha * dy;
  PetscInt i = shape.s_p(x, y, z);
  PetscInt j = indexing::j_p(x, z);

  PetscReal wy_p = -qy * (shape(i, New, Y) - shape(i, Old, Y)) *
    (shape(i, New, X) * (2.0 * shape(i, New, Z) + shape(i, Old, Z)) +
      shape(i, Old, X) * (2.0 * shape(i, Old, Z) + shape(i, New, Z)));

  temp_jy[j] = (static_cast<PetscReal>(y > 0) * temp_jy[j]) + wy_p;
  PetscFunctionReturn(temp_jy[j]);
}

PetscReal EsirkepovDecomposition::get_jz(
  PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jz) const
{
  PetscFunctionBeginHot;
  PetscReal qz = alpha * dz;
  PetscInt i = shape.s_p(x, y, z);
  PetscInt j = indexing::j_p(x, y);

  PetscReal wz_p = -qz * (shape(i, New, Z) - shape(i, Old, Z)) *
    (shape(i, New, Y) * (2.0 * shape(i, New, X) + shape(i, Old, X)) +
      shape(i, Old, Y) * (2.0 * shape(i, Old, X) + shape(i, New, X)));

  temp_jz[j] = (static_cast<PetscReal>(z > 0) * temp_jz[j]) + wz_p;
  PetscFunctionReturn(temp_jz[j]);
}
