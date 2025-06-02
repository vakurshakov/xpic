#include "implicit_esirkepov.h"

#include "src/utils/shape.h"

ImplicitEsirkepov::ImplicitEsirkepov(
  Vector3R*** E_g, Vector3R*** B_g, Vector3R*** J_g)
  : E_g(E_g), B_g(B_g), J_g(J_g)
{
}

void ImplicitEsirkepov::Shape::setup(
  const Vector3R& rn, const Vector3R& r0, Type t)
{
  Vector3R p_rn = ::Shape::make_r(rn);
  Vector3R p_r0 = ::Shape::make_r(r0);

  start = ::Shape::make_start(min(p_r0, p_rn), shr2);
  size = ::Shape::make_end(max(p_r0, p_rn), shr2);
  size -= start;

  Vector3R sn, s0, sh;

  static constexpr PetscReal alpha = 1.0 / 6.0;

  for (PetscInt i = 0; i < size.elements_product(); ++i) {
    auto gc_x = (PetscReal)(start[X] + i % size[X]);
    auto gc_y = (PetscReal)(start[Y] + (i / size[X]) % size[Y]);
    auto gc_z = (PetscReal)(start[Z] + (i / size[X]) / size[Y]);

    auto gv_x = gc_x + 0.5;
    auto gv_y = gc_y + 0.5;
    auto gv_z = gc_z + 0.5;

    if (t == magnetic) {
      std::swap(gc_x, gv_x);
      std::swap(gc_y, gv_y);
      std::swap(gc_z, gv_z);
    }

    sn[X] = sfunc2(gc_x - p_rn[X]);
    sn[Y] = sfunc2(gc_y - p_rn[Y]);
    sn[Z] = sfunc2(gc_z - p_rn[Z]);

    s0[X] = sfunc2(gc_x - p_r0[X]);
    s0[Y] = sfunc2(gc_y - p_r0[Y]);
    s0[Z] = sfunc2(gc_z - p_r0[Z]);

    sh[X] = alpha * sfunc1(gv_x - 0.5 * (p_rn[X] + p_r0[X]));
    sh[Y] = alpha * sfunc1(gv_y - 0.5 * (p_rn[Y] + p_r0[Y]));
    sh[Z] = alpha * sfunc1(gv_z - 0.5 * (p_rn[Z] + p_r0[Z]));

    // clang-format off
    cache[i][X] = sh[X] * (sn[Y] * (2.0 * sn[Z] + s0[Z]) + s0[Y] * (2.0 * s0[Z] + sn[Z]));
    cache[i][Y] = sh[Y] * (sn[Z] * (2.0 * sn[X] + s0[X]) + s0[Z] * (2.0 * s0[X] + sn[X]));
    cache[i][Z] = sh[Z] * (sn[X] * (2.0 * sn[Y] + s0[Y]) + s0[X] * (2.0 * s0[Y] + sn[Y]));
    // clang-format on
  }
}


void ImplicitEsirkepov::interpolate(
  Vector3R& E_p, Vector3R& B_p, const Vector3R& rn, const Vector3R& r0)
{
  shape[0].setup(rn, r0, electric);
  shape[1].setup(rn, r0, magnetic);

  auto& sh = shape[0];

  for (PetscInt i = 0; i < sh.size.elements_product(); ++i) {
    PetscInt g_x = sh.start[X] + i % sh.size[X];
    PetscInt g_y = sh.start[Y] + (i / sh.size[X]) % sh.size[Y];
    PetscInt g_z = sh.start[Z] + (i / sh.size[X]) / sh.size[Y];

    E_p += E_g[g_z][g_y][g_x].elementwise_product(shape[electric].cache[i]);
    B_p += B_g[g_z][g_y][g_x].elementwise_product(shape[magnetic].cache[i]);
  }
}

void ImplicitEsirkepov::decompose(
  PetscReal alpha, const Vector3R& v, const Vector3R& rn, const Vector3R& r0)
{
  auto& sh = shape[0];
  sh.setup(rn, r0, electric);

  for (PetscInt i = 0; i < sh.size.elements_product(); ++i) {
    PetscInt g_x = sh.start[X] + i % sh.size[X];
    PetscInt g_y = sh.start[Y] + (i / sh.size[X]) % sh.size[Y];
    PetscInt g_z = sh.start[Z] + (i / sh.size[X]) / sh.size[Y];

    Vector3R J_p = alpha * v.elementwise_product(shape[electric].cache[i]);

#pragma omp atomic
    J_g[g_z][g_y][g_x][X] += J_p[X];

#pragma omp atomic
    J_g[g_z][g_y][g_x][Y] += J_p[Y];

#pragma omp atomic
    J_g[g_z][g_y][g_x][Z] += J_p[Z];
  }
}
