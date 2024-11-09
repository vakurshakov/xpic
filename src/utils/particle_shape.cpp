#include "particle_shape.h"

Node::Node(const Vector3R& pr)
{
  r = Node::make_r(pr);
  g = Node::make_g(r);
}

/* static */ Vector3R Node::make_r(const Vector3R& pr)
{
  return {
    pr.x() / dx,
    pr.y() / dy,
    pr.z() / dz,
  };
}

/* static */ Vector3I Node::make_g(const Vector3R& nr)
{
  return {
    ROUND(nr.x()) - shape_radius,
    ROUND(nr.y()) - shape_radius,
    ROUND(nr.z()) - shape_radius,
  };
}

void Shape::fill(const Vector3I& p_g, const Vector3R& p_r, bool shift,
  PetscReal (&sfunc)(PetscReal), PetscInt width)
{
  PetscReal g_x, g_y, g_z;

#pragma omp simd collapse(Vector3I::dim)
  // clang-format off
  for (PetscInt z = 0; z < width; ++z) {
  for (PetscInt y = 0; y < width; ++y) {
  for (PetscInt x = 0; x < width; ++x) {
    g_x = p_g[X] + x;
    g_y = p_g[Y] + y;
    g_z = p_g[Z] + z;

    if (shift) {
      g_x += 0.5;
      g_y += 0.5;
      g_z += 0.5;
    }

    PetscInt i = Shape::index(z, y, x);
    shape[i * 3 + X] = sfunc(p_r[X] - g_x);
    shape[i * 3 + Y] = sfunc(p_r[Y] - g_y);
    shape[i * 3 + Z] = sfunc(p_r[Z] - g_z);
  }}}
  // clang-format on
}
