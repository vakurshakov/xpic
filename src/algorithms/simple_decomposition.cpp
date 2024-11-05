#include "simple_decomposition.h"

Simple_decomposition::Simple_decomposition(
  const Vector3I& width, const Vector3R& p_j, const Shape& no, const Shape& sh)
  : width(width), J_p(p_j), no(no), sh(sh)
{
}

void Simple_decomposition::process(const Vector3R& p_r, Context& J) const
{
  Vector3I p_gc = Node::make_g(p_r);
  Vector3I p_go = Node::make_g(p_r + Vector3R{0.5});

  PetscInt gc_x, gc_y, gc_z;
  PetscInt go_x, go_y, go_z;

  // clang-format off: @todo create macro/range-based analogue for this loop
  for (PetscInt z = 0; z < width[Z]; ++z) {
  for (PetscInt y = 0; y < width[Y]; ++y) {
  for (PetscInt x = 0; x < width[X]; ++x) {
    PetscInt i = Shape::index(x, y, z);

    Vector3R J_shape = {
      no(i, Z) * no(i, Y) * sh(i, X),
      no(i, Z) * sh(i, Y) * no(i, X),
      sh(i, Z) * no(i, Y) * no(i, X),
    };

    gc_x = p_gc[X] + x;
    gc_y = p_gc[Y] + y;
    gc_z = p_gc[Z] + z;

    go_x = p_go[X] + x;
    go_y = p_go[Y] + y;
    go_z = p_go[Z] + z;

#pragma omp atomic update
      J[gc_z][gc_y][go_x][X] += J_p.x() * J_shape.x();

#pragma omp atomic update
      J[gc_z][go_y][gc_x][Y] += J_p.y() * J_shape.y();

#pragma omp atomic update
      J[go_z][gc_y][gc_x][Z] += J_p.z() * J_shape.z();
  }}}
  // clang-format on
}
