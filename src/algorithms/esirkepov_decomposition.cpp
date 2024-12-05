#include "esirkepov_decomposition.h"

namespace indexing {

constexpr PetscInt j_p(PetscInt x1, PetscInt x2)
{
  return x1 * shape_width + x2;
}

}  // namespace indexing


EsirkepovDecomposition::EsirkepovDecomposition(const Shape& shape, PetscReal alpha)
  : shape(shape), alpha(alpha)
{
}

PetscErrorCode EsirkepovDecomposition::process(Context& J) const
{
  PetscFunctionBeginHot;
  static constexpr PetscInt j_geom = POW2(shape_width);
  static constexpr PetscInt j_comp = Vector3I::dim;
  static constexpr std::size_t j_size = static_cast<std::size_t>(j_geom) * j_comp;

  std::array<PetscReal, j_size> temp_j;

  // clang-format off: @todo create macro/range-based analogue for this loop
  for (PetscInt z = 0; z < shape.size[Z]; ++z) {
  for (PetscInt y = 0; y < shape.size[Y]; ++y) {
  for (PetscInt x = 0; x < shape.size[X]; ++x) {
    PetscInt g_x = shape.start[X] + x;
    PetscInt g_y = shape.start[Y] + y;
    PetscInt g_z = shape.start[Z] + z;

    PetscReal p_jx = get_jx(z, y, x, temp_j.data() + static_cast<std::ptrdiff_t>(j_geom * X));
    PetscReal p_jy = get_jy(z, y, x, temp_j.data() + static_cast<std::ptrdiff_t>(j_geom * Y));
    PetscReal p_jz = get_jz(z, y, x, temp_j.data() + static_cast<std::ptrdiff_t>(j_geom * Z));

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


PetscReal EsirkepovDecomposition::get_jx(
  PetscInt z, PetscInt y, PetscInt x, PetscReal* temp_jx) const
{
  PetscFunctionBeginHot;
  PetscReal qx = alpha * dx;
  PetscInt i = shape.s_p(z, y, x);
  PetscInt j = indexing::j_p(z, y);

  PetscReal wx_p = -qx * (shape(i, New, X) - shape(i, Old, X)) *
    (shape(i, New, Y) * (2.0 * shape(i, New, Z) + shape(i, Old, Z)) +
      shape(i, Old, Y) * (2.0 * shape(i, Old, Z) + shape(i, New, Z)));

  temp_jx[j] = (static_cast<PetscReal>(x > 0) * temp_jx[j]) + wx_p;
  PetscFunctionReturn(temp_jx[j]);
}

PetscReal EsirkepovDecomposition::get_jy(
  PetscInt z, PetscInt y, PetscInt x, PetscReal* temp_jy) const
{
  PetscFunctionBeginHot;
  PetscReal qy = alpha * dy;
  PetscInt i = shape.s_p(z, y, x);
  PetscInt j = indexing::j_p(z, x);

  PetscReal wy_p = -qy * (shape(i, New, Y) - shape(i, Old, Y)) *
    (shape(i, New, X) * (2.0 * shape(i, New, Z) + shape(i, Old, Z)) +
      shape(i, Old, X) * (2.0 * shape(i, Old, Z) + shape(i, New, Z)));

  temp_jy[j] = (static_cast<PetscReal>(y > 0) * temp_jy[j]) + wy_p;
  PetscFunctionReturn(temp_jy[j]);
}

PetscReal EsirkepovDecomposition::get_jz(
  PetscInt z, PetscInt y, PetscInt x, PetscReal* temp_jz) const
{
  PetscFunctionBeginHot;
  PetscReal qz = alpha * dz;
  PetscInt i = shape.s_p(z, y, x);
  PetscInt j = indexing::j_p(y, x);

  PetscReal wz_p = -qz * (shape(i, New, Z) - shape(i, Old, Z)) *
    (shape(i, New, Y) * (2.0 * shape(i, New, X) + shape(i, Old, X)) +
      shape(i, Old, Y) * (2.0 * shape(i, Old, X) + shape(i, New, X)));

  temp_jz[j] = (static_cast<PetscReal>(z > 0) * temp_jz[j]) + wz_p;
  PetscFunctionReturn(temp_jz[j]);
}
