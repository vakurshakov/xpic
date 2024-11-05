#include "particle_shape.h"

Node::Node(const Vector3R& __r)
{
  r = {
    __r.x() / dx,
    __r.y() / dy,
    __r.z() / dz,
  };

  g = {
    (geom_nx > 1) ? (PetscInt)std::ceil(r.x()) - shape_radius : 0,
    (geom_ny > 1) ? (PetscInt)std::ceil(r.y()) - shape_radius : 0,
    (geom_nz > 1) ? (PetscInt)std::ceil(r.z()) - shape_radius : 0,
  };
}

PetscErrorCode fill_shape(const Vector3I& p_g, const Vector3R& p_r,
  const Vector3I& l_width, bool shift, Shape& shape)
{
  PetscFunctionBeginHot;
  PetscReal g_x, g_y, g_z;

#pragma omp simd collapse(Vector3I::dim)
  // clang-format off
  for (PetscInt z = 0; z < l_width[Z]; ++z) {
  for (PetscInt y = 0; y < l_width[Y]; ++y) {
  for (PetscInt x = 0; x < l_width[X]; ++x) {
    g_x = p_g[X] + x;
    g_y = p_g[Y] + y;
    g_z = p_g[Z] + z;

    if (shift) {
      g_x += 0.5;
      g_y += 0.5;
      g_z += 0.5;
    }

    PetscInt i = Shape::index(x, y, z);
    shape(i, X) = shape_function(p_r[X] - g_x, X);
    shape(i, Y) = shape_function(p_r[Y] - g_y, Y);
    shape(i, Z) = shape_function(p_r[Z] - g_z, Z);
  }}}
  // clang-format on
  PetscFunctionReturn(PETSC_SUCCESS);
}
