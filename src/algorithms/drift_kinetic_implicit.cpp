#include "drift_kinetic_implicit.h"

#include "src/algorithms/simple_interpolation.h"

constexpr auto& sfunc0 = spline_of_0th_order;
constexpr auto& sfunc1 = spline_of_1st_order;
constexpr auto& sfunc2 = spline_of_2nd_order;

constexpr auto sfunc11 = 
[](PetscReal left_s, PetscReal right_s) {
  return sfunc1(left_s)*sfunc1(right_s);
};

constexpr auto sfuncB = 
[](PetscReal left_s0, PetscReal left_sn, PetscReal right_s0, PetscReal right_sn) {
  return sfunc11(left_s0, right_sn) + sfunc11(left_sn, right_s0) +
           2 * (sfunc11(left_s0, right_s0) + sfunc11(left_sn, right_sn));
};

constexpr auto sfunc22 = 
[](PetscReal left_s, PetscReal right_s) {
  return sfunc2(left_s)*sfunc2(right_s);
};

constexpr auto sfuncE = 
[](PetscReal left_s0, PetscReal left_sn, PetscReal right_s0, PetscReal right_sn) {
  return sfunc22(left_s0, right_sn) + sfunc22(left_sn, right_s0) +
           2 * (sfunc22(left_s0, right_s0) + sfunc22(left_sn, right_sn));
};

DriftKineticEsirkepov::DriftKineticEsirkepov(
  Vector3R*** E_g, Vector3R*** B_g, Vector3R*** J_g, Vector3R*** gradB_g)
  : E_g(E_g), B_g(B_g), J_g(J_g), gradB_g(gradB_g)
{
}

PetscErrorCode DriftKineticEsirkepov::set_dBidrj(
  Vector3R*** _dBdx_g, Vector3R*** _dBdy_g, Vector3R*** _dBdz_g)
{
  PetscFunctionBeginHot;
  dBidx_g = _dBdx_g;
  dBidy_g = _dBdy_g;
  dBidz_g = _dBdz_g;

  PetscReal inv_dx = 1.0 / dx, inv_dy = 1.0 / dy, inv_dz = 1.0 / dz;

  PetscInt ip, jp, kp;

  for (PetscInt k = 0; k < geom_nz; ++k) {
    for (PetscInt j = 0; j < geom_ny; ++j) {
      for (PetscInt i = 0; i < geom_nx; ++i) {
        ip = i + 1;
        jp = j + 1;
        kp = k + 1;

        const auto& B_center = B_g[k][j][i];
        dBidx_g[k][j][i] = (B_g[k][j][ip] - B_center) * inv_dx;
        dBidy_g[k][j][i] = (B_g[k][jp][i] - B_center) * inv_dy;
        dBidz_g[k][j][i] = (B_g[kp][j][i] - B_center) * inv_dz;

        std::cout << ip+jp+kp << " " << dBidz_g[k][j][i][X] << " " << dBidz_g[k][j][i][Y] << " " << dBidz_g[k][j][i][Z] << std::endl;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::interpolate_E(Vector3R& E_p, const Vector3R& rn, const Vector3R& r0)
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
        sfuncE(g_y - p_r0[Y], g_y - p_rn[Y], g_z - p_r0[Z], g_z - p_rn[Z]);

    E_p[Y] += alpha * E[Y] * sfunc1(g_y + 0.5 - 0.5 * (p_rn[Y] + p_r0[Y])) *
        sfuncE(g_z - p_r0[Z], g_z - p_rn[Z], g_x - p_r0[X], g_x - p_rn[X]);

    E_p[Z] += alpha * E[Z] * sfunc1(g_z + 0.5 - 0.5 * (p_rn[Z] + p_r0[Z])) *
        sfuncE(g_x - p_r0[X], g_x - p_rn[X], g_y - p_r0[Y], g_y - p_rn[Y]);
    //E_p[X] += alpha * E[X] * sfunc1(g_x + 0.5 - 0.5 * (p_rn[X] + p_r0[X])) *
    //   (sfunc2(g_y - p_rn[Y]) * (2.0 * sfunc2(g_z - p_rn[Z]) + sfunc2(g_z - p_r0[Z])) +
    //     sfunc2(g_y - p_r0[Y]) * (2.0 * sfunc2(g_z - p_r0[Z]) + sfunc2(g_z - p_rn[Z])));
//
    //E_p[Y] += alpha * E[Y] * sfunc1(g_y + 0.5 - 0.5 * (p_rn[Y] + p_r0[Y])) *
    //   (sfunc2(g_z - p_rn[Z]) * (2.0 * sfunc2(g_x - p_rn[X]) + sfunc2(g_x - p_r0[X])) +
    //     sfunc2(g_z - p_r0[Z]) * (2.0 * sfunc2(g_x - p_r0[X]) + sfunc2(g_x - p_rn[X])));
//
    //E_p[Z] += alpha * E[Z] * sfunc1(g_z + 0.5 - 0.5 * (p_rn[Z] + p_r0[Z])) *
    //   (sfunc2(g_x - p_rn[X]) * (2.0 * sfunc2(g_y - p_rn[Y]) + sfunc2(g_y - p_r0[Y])) +
    //     sfunc2(g_x - p_r0[X]) * (2.0 * sfunc2(g_y - p_r0[Y]) + sfunc2(g_y - p_rn[Y])));
    // clang-format on
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::interpolate_B(Vector3R& B_p, const Vector3R& rn) {
  PetscFunctionBeginHot;

  ::Shape shape;
  shape.setup(rn, 1., sfunc1);

  SimpleInterpolation interpolation(shape);
  SimpleInterpolation::Context E_dummy;
  SimpleInterpolation::Context B_fields{{B_p, B_g}};
  interpolation.process(E_dummy, B_fields);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
PetscErrorCode DriftKineticEsirkepov::interpolate_gradB(Vector3R& gradB_p, Vector3R& b_p, const Vector3R& rn, const Vector3R& r0){
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
    (PetscInt)std::round(p_rn[X]) - 1,
    (PetscInt)std::round(p_rn[Y]) - 1,
    (PetscInt)std::round(p_rn[Z]) - 1,
  };

  PetscReal alpha = 1.0 / 6.0;

  PetscInt shw = 2 * 1 + 1;

  for (PetscInt i = 0; i < POW3(shw); ++i) {
    PetscInt g_x = p_g[X] + i % shw;
    PetscInt g_y = p_g[Y] + (i / shw) % shw;
    PetscInt g_z = p_g[Z] + (i / shw) / shw;

    // clang-format off
    // clang-format on
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
*/

PetscErrorCode DriftKineticEsirkepov::interpolate(Vector3R& E_p, Vector3R& B_p,
  Vector3R& gradB_p, const Vector3R& rn, const Vector3R& r0)
{
  PetscFunctionBeginHot;
  E_p = {};
  B_p = {};
  gradB_p = {};

  interpolate_E(E_p, rn, r0);
  interpolate_B(B_p, rn);
  //interpolate_gradB(gradB_p, B_p.normalized(), rn, r0);

  ::Shape shape;
  shape.setup(rn,1.0,sfunc1);
  SimpleInterpolation interpolation(shape);
  SimpleInterpolation::Context e_fields;
  SimpleInterpolation::Context gradb_fields;
  if (gradB_g != nullptr) {
    gradb_fields.emplace_back(gradB_p, gradB_g);
  }
  interpolation.process(e_fields, gradb_fields);
  PetscFunctionReturn(PETSC_SUCCESS);
}
