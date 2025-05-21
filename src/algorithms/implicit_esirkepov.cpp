#include "implicit_esirkepov.h"

#include "src/utils/vector_utils.h"

namespace ImplicitEsirkepov {

constexpr auto& sfunc1 = spline_of_1st_order;
constexpr auto& sfunc2 = spline_of_2nd_order;

PetscErrorCode interpolation(
  Vector3R& E_p, Vector3R*** E_g, const Vector3R& rn, const Vector3R& r0)
{
  PetscFunctionBeginHot;
  Vector3R p_rn{
    rn[X] / dx,
    rn[Y] / dy,
    rn[Z] / dz,
  };

  Vector3R p_r0{
    r0[X] / dx,
    r0[Y] / dy,
    r0[Z] / dz,
  };

  Vector3I p_g{
    (PetscInt)std::round(p_rn[X]) - 2,
    (PetscInt)std::round(p_rn[Y]) - 2,
    (PetscInt)std::round(p_rn[Z]) - 2,
  };

  PetscReal alpha = 1.0 / 6.0;

  PetscInt shw = 2 * 2 + 1;

  for (PetscInt i = 0; i < POW3(shw); ++i) {
    PetscInt g_x = p_g[X] + i % shw;
    PetscInt g_y = p_g[Y] + (i / shw) % shw;
    PetscInt g_z = p_g[Z] + (i / shw) / shw;

    auto& E = E_g[g_z][g_y][g_x];

    // clang-format off
    E_p[X] += alpha * E[X] * sfunc1(g_x + 0.5 - 0.5 * (p_rn[X] + p_r0[X])) *
       (sfunc2(g_y - p_rn[Y]) * (2.0 * sfunc2(g_z - p_rn[Z]) + sfunc2(g_z - p_r0[Z])) +
         sfunc2(g_y - p_r0[Y]) * (2.0 * sfunc2(g_z - p_r0[Z]) + sfunc2(g_z - p_rn[Z])));

    E_p[Y] += alpha * E[Y] * sfunc1(g_y + 0.5 - 0.5 * (p_rn[Y] + p_r0[Y])) *
       (sfunc2(g_z - p_rn[Z]) * (2.0 * sfunc2(g_x - p_rn[X]) + sfunc2(g_x - p_r0[X])) +
         sfunc2(g_z - p_r0[Z]) * (2.0 * sfunc2(g_x - p_r0[X]) + sfunc2(g_x - p_rn[X])));

    E_p[Z] += alpha * E[Z] * sfunc1(g_z + 0.5 - 0.5 * (p_rn[Z] + p_r0[Z])) *
       (sfunc2(g_x - p_rn[X]) * (2.0 * sfunc2(g_y - p_rn[Y]) + sfunc2(g_y - p_r0[Y])) +
         sfunc2(g_x - p_r0[X]) * (2.0 * sfunc2(g_y - p_r0[Y]) + sfunc2(g_y - p_rn[Y])));
    // clang-format on
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode decomposition(Vector3R*** J_g, //
  const Vector3R& rn, const Vector3R& r0, const Vector3R& v, PetscReal alpha)
{
  PetscFunctionBeginHot;
  Vector3R p_rn{
    rn[X] / dx,
    rn[Y] / dy,
    rn[Z] / dz,
  };

  Vector3R p_r0{
    r0[X] / dx,
    r0[Y] / dy,
    r0[Z] / dz,
  };

  PetscInt shr = 2;
  PetscInt shw = 2 * shr + 1;

  Vector3I p_g{
    (PetscInt)std::floor(p_rn[X]) - shr,
    (PetscInt)std::floor(p_rn[Y]) - shr,
    (PetscInt)std::floor(p_rn[Z]) - shr,
  };

  for (PetscInt i = 0; i < POW3(shw); ++i) {
    PetscInt g_x = p_g[X] + i % shw;
    PetscInt g_y = p_g[Y] + (i / shw) % shw;
    PetscInt g_z = p_g[Z] + (i / shw) / shw;

    Vector3R J_p;

    // clang-format off
    J_p[X] = alpha * v[X] * sfunc1(g_x + 0.5 - 0.5 * (p_rn[X] + p_r0[X])) *
       (sfunc2(g_y - p_rn[Y]) * (2.0 * sfunc2(g_z - p_rn[Z]) + sfunc2(g_z - p_r0[Z])) +
         sfunc2(g_y - p_r0[Y]) * (2.0 * sfunc2(g_z - p_r0[Z]) + sfunc2(g_z - p_rn[Z])));

    J_p[Y] = alpha * v[Y] * sfunc1(g_y + 0.5 - 0.5 * (p_rn[Y] + p_r0[Y])) *
       (sfunc2(g_z - p_rn[Z]) * (2.0 * sfunc2(g_x - p_rn[X]) + sfunc2(g_x - p_r0[X])) +
         sfunc2(g_z - p_r0[Z]) * (2.0 * sfunc2(g_x - p_r0[X]) + sfunc2(g_x - p_rn[X])));

    J_p[Z] = alpha * v[Z] * sfunc1(g_z + 0.5 - 0.5 * (p_rn[Z] + p_r0[Z])) *
       (sfunc2(g_x - p_rn[X]) * (2.0 * sfunc2(g_y - p_rn[Y]) + sfunc2(g_y - p_r0[Y])) +
         sfunc2(g_x - p_r0[X]) * (2.0 * sfunc2(g_y - p_r0[Y]) + sfunc2(g_y - p_rn[Y])));
    // clang-format on

    // LOG("r0: {: 5.3f} {: 5.3f} {: 5.3f}, rn: {: 5.3f} {: 5.3f} {: 5.3f}, g: {: 3d} {: 3d} {: 3d}, i: {: 2d} {: 2d} {: 2d} Jp[X]: {: 5.3e}", REP3_A(r0), REP3_A(rn), REP3_X(g_), i % shw, (i / shw) % shw, (i / shw) / shw, J_p[X]);

#pragma omp atomic
    J_g[g_z][g_y][g_x][X] += J_p[X];

#pragma omp atomic
    J_g[g_z][g_y][g_x][Y] += J_p[Y];

#pragma omp atomic
    J_g[g_z][g_y][g_x][Z] += J_p[Z];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace ImplicitEsirkepov
