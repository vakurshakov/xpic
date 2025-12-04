#include "drift_kinetic_implicit.h"

#include "src/algorithms/simple_interpolation.h"

PetscReal NGP_left(PetscReal s)
{
  if (s < 0.5 && s >= -0.5)
    return 1.0;
  return 0.0;
}

constexpr auto& sfunc0 = NGP_left;
constexpr auto& sfunc1 = spline_of_1st_order;
constexpr auto& sfunc2 = spline_of_2nd_order;

constexpr PetscReal alpha = 1./6.;

constexpr auto alongSBlong = 
[](PetscReal left_coeff, PetscReal left_s, PetscReal right_coeff, PetscReal right_s) {
  return alpha * (left_coeff * sfunc0(left_s) - right_coeff * sfunc0(right_s));
};

constexpr auto alongSBshort = 
[](PetscReal left_coeff, PetscReal left_s, PetscReal right_coeff, PetscReal right_s) {
   left_coeff = 1.,right_coeff = 0.;
  return alongSBlong(left_coeff, left_s, right_coeff, right_s);
};

constexpr auto sfunc11 = 
[](PetscReal left_s, PetscReal right_s) {
  return sfunc1(left_s)*sfunc1(right_s);
};

constexpr auto acrossSBfunc = 
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


DriftKineticEsirkepov::DriftKineticEsirkepov(
  Vector3R*** E_g, Vector3R*** B_g, Vector3R*** J_g, Vector3R*** M_g, //
  Vector3R*** dBidx_g, Vector3R*** dBidy_g, Vector3R*** dBidz_g)
  : E_g(E_g), B_g(B_g), J_g(J_g), M_g(M_g)
{
  set_dBidrj(dBidx_g, dBidy_g, dBidz_g);
}

PetscErrorCode DriftKineticEsirkepov::set_dBidrj(
  Vector3R*** _dBidx_g, Vector3R*** _dBidy_g, Vector3R*** _dBidz_g)
{
  PetscFunctionBeginHot;
  dBidx_g = _dBidx_g;
  dBidy_g = _dBidy_g;
  dBidz_g = _dBidz_g;

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

      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::interpolate_E
(Vector3R& E_p, const Vector3R& Rsn, const Vector3R& Rs0)
{
  PetscFunctionBeginHot;

  Vector3R p_Rsn{
    Rsn[X] / dx,
    Rsn[Y] / dy,
    Rsn[Z] / dz,
  };

  Vector3R p_Rs0{
    Rs0[X] / dx,
    Rs0[Y] / dy,
    Rs0[Z] / dz,
  };

  Vector3R p_dR{
    p_Rsn[X] != p_Rs0[X] ? p_Rsn[X] - p_Rs0[X] : 1.0,
    p_Rsn[Y] != p_Rs0[Y] ? p_Rsn[Y] - p_Rs0[Y] : 1.0,
    p_Rsn[Z] != p_Rs0[Z] ? p_Rsn[Z] - p_Rs0[Z] : 1.0,
  };

  PetscInt shr = 2;
  Vector3I p_g{
    (PetscInt)std::round(p_Rsn[X]) - shr,
    (PetscInt)std::round(p_Rsn[Y]) - shr,
    (PetscInt)std::round(p_Rsn[Z]) - shr,
  };

  PetscInt shw = 2 * shr + 1;

  for (PetscInt i = 0; i < POW3(shw); ++i) {
    PetscInt g_x = p_g[X] + i % shw;
    PetscInt g_y = p_g[Y] + (i / shw) % shw;
    PetscInt g_z = p_g[Z] + (i / shw) / shw;

    auto& E = E_g[g_z][g_y][g_x];

    // clang-format off
    E_p[X] += alpha * E[X] * p_dR[X] * sfunc1(g_x + 0.5 - 0.5 * (p_Rsn[X] + p_Rs0[X])) *
        sfuncE(g_y - p_Rs0[Y], g_y - p_Rsn[Y], g_z - p_Rs0[Z], g_z - p_Rsn[Z]);

    E_p[Y] += alpha * E[Y] * p_dR[Y] * sfunc1(g_y + 0.5 - 0.5 * (p_Rsn[Y] + p_Rs0[Y])) *
        sfuncE(g_z - p_Rs0[Z], g_z - p_Rsn[Z], g_x - p_Rs0[X], g_x - p_Rsn[X]);

    E_p[Z] += alpha * E[Z] * p_dR[Z] * sfunc1(g_z + 0.5 - 0.5 * (p_Rsn[Z] + p_Rs0[Z])) *
        sfuncE(g_x - p_Rs0[X], g_x - p_Rsn[X], g_y - p_Rs0[Y], g_y - p_Rsn[Y]);
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

PetscErrorCode DriftKineticEsirkepov::interpolate_gradB
(Vector3R& gradB_p, Vector3R& b_p, const Vector3R& Rsn, const Vector3R& Rs0){
  PetscFunctionBeginHot;

  Vector3R p_Rsn{
    Rsn[X] / dx,
    Rsn[Y] / dy,
    Rsn[Z] / dz,
  };

  Vector3R p_Rs0{
    Rs0[X] / dx,
    Rs0[Y] / dy,
    Rs0[Z] / dz,
  };

  Vector3R p_dR{
    p_Rsn[X] - p_Rs0[X],
    p_Rsn[Y] - p_Rs0[Y],
    p_Rsn[Z] - p_Rs0[Z],
  };

  using AlongSBFunc = PetscReal (*)(PetscReal, PetscReal, PetscReal, PetscReal);
  AlongSBFunc alongSBx = alongSBlong, alongSBy = alongSBlong, alongSBz = alongSBlong;

  if ((std::abs(p_Rsn[X] - p_Rs0[X]) < PETSC_MACHINE_EPSILON)) {alongSBx = alongSBshort;}
  if ((std::abs(p_Rsn[Y] - p_Rs0[Y]) < PETSC_MACHINE_EPSILON)) {alongSBy = alongSBshort;}
  if ((std::abs(p_Rsn[Z] - p_Rs0[Z]) < PETSC_MACHINE_EPSILON)) {alongSBz = alongSBshort;}

  Vector3R p_Rsmid{
    0.5 * (p_Rsn[X] + p_Rs0[X]),
    0.5 * (p_Rsn[Y] + p_Rs0[Y]),
    0.5 * (p_Rsn[Z] + p_Rs0[Z]),
  };

  Vector3I p_gs{
    (PetscInt)std::round(p_Rsmid[X]),
    (PetscInt)std::round(p_Rsmid[Y]),
    (PetscInt)std::round(p_Rsmid[Z]),
  };

  PetscInt shr = 2;
  Vector3I p_g{
    (PetscInt)std::round(p_Rsmid[X]) - shr,
    (PetscInt)std::round(p_Rsmid[Y]) - shr,
    (PetscInt)std::round(p_Rsmid[Z]) - shr,
  };

  PetscInt shw = 2 * shr + 1;

  for (PetscInt i = 0; i < POW3(shw); ++i) {
    PetscInt g_x = p_g[X] + i % shw;
    PetscInt g_y = p_g[Y] + (i / shw) % shw;
    PetscInt g_z = p_g[Z] + (i / shw) / shw;

    auto& dBidx = dBidx_g[g_z][g_y][g_x];
    auto& dBidy = dBidy_g[g_z][g_y][g_x];
    auto& dBidz = dBidz_g[g_z][g_y][g_x];

    PetscReal Sxx = alongSBx(p_dR[X], g_x + 0.5 - p_Rsmid[X], 0., 0.) *
                    acrossSBfunc(g_y + 0.5 - p_Rs0[Y], g_y + 0.5 - p_Rsn[Y],
                           g_z + 0.5 - p_Rs0[Z], g_z + 0.5 - p_Rsn[Z]); 
    PetscReal Sxy = alongSBx(p_gs[X] + 0.5 - p_Rs0[X], g_x - 0.5 * (p_gs[X] + 0.5 + p_Rs0[X]),
                             p_gs[X] + 0.5 - p_Rsn[X], g_x - 0.5 * (p_gs[X] + 0.5 + p_Rsn[X])) *
                    acrossSBfunc(g_y - p_Rs0[Y], g_y - p_Rsn[Y],
                                 g_z + 0.5 - p_Rs0[Z], g_z + 0.5 - p_Rsn[Z]);
    PetscReal Sxz = alongSBx(p_gs[X] + 0.5 - p_Rs0[X], g_x - 0.5 * (p_gs[X] + 0.5 + p_Rs0[X]),
                             p_gs[X] + 0.5 - p_Rsn[X], g_x - 0.5 * (p_gs[X] + 0.5 + p_Rsn[X])) *
                    acrossSBfunc(g_y + 0.5 - p_Rs0[Y], g_y + 0.5 - p_Rsn[Y],
                                 g_z - p_Rs0[Z], g_z - p_Rsn[Z]);

    PetscReal Syy = alongSBy(p_dR[Y], g_y + 0.5 - p_Rsmid[Y], 0., 0.) *
                    acrossSBfunc(g_z + 0.5 - p_Rs0[Z], g_z + 0.5 - p_Rsn[Z],
                           g_x + 0.5 - p_Rs0[X], g_x + 0.5 - p_Rsn[X]);
    PetscReal Syx = alongSBy(p_gs[Y] + 0.5 - p_Rs0[Y], g_y - 0.5 * (p_gs[Y] + 0.5 + p_Rs0[Y]),
                             p_gs[Y] + 0.5 - p_Rsn[Y], g_y - 0.5 * (p_gs[Y] + 0.5 + p_Rsn[Y])) *
                    acrossSBfunc(g_z + 0.5 - p_Rs0[Z], g_z + 0.5 - p_Rsn[Z],
                           g_x - p_Rs0[X], g_x - p_Rsn[X]);
    PetscReal Syz = alongSBy(p_gs[Y] + 0.5 - p_Rs0[Y], g_y - 0.5 * (p_gs[Y] + 0.5 + p_Rs0[Y]),
                             p_gs[Y] + 0.5 - p_Rsn[Y], g_y - 0.5 * (p_gs[Y] + 0.5 + p_Rsn[Y])) *
                    acrossSBfunc(g_z - p_Rs0[Z], g_z - p_Rsn[Z],
                           g_x + 0.5 - p_Rs0[X], g_x + 0.5 - p_Rsn[X]);
                    
    PetscReal Szz = alongSBz(p_dR[Z], g_z + 0.5 - p_Rsmid[Z], 0., 0.) *
                    acrossSBfunc(g_x + 0.5 - p_Rs0[X], g_x + 0.5 - p_Rsn[X],
                           g_y + 0.5 - p_Rs0[Y], g_y + 0.5 - p_Rsn[Y]);
    PetscReal Szx = alongSBz(p_gs[Z] + 0.5 - p_Rs0[Z], g_z - 0.5 * (p_gs[Z] + 0.5 + p_Rs0[Z]),
                             p_gs[Z] + 0.5 - p_Rsn[Z], g_z - 0.5 * (p_gs[Z] + 0.5 + p_Rsn[Z])) *
                    acrossSBfunc(g_x - p_Rs0[X], g_x - p_Rsn[X],
                           g_y + 0.5 - p_Rs0[Y], g_y + 0.5 - p_Rsn[Y]);
    PetscReal Szy = alongSBz(p_gs[Z] + 0.5 - p_Rs0[Z], g_z - 0.5 * (p_gs[Z] + 0.5 + p_Rs0[Z]),
                             p_gs[Z] + 0.5 - p_Rsn[Z], g_z - 0.5 * (p_gs[Z] + 0.5 + p_Rsn[Z])) *
                    acrossSBfunc(g_x + 0.5 - p_Rs0[X], g_x + 0.5 - p_Rsn[X],
                           g_y - p_Rs0[Y], g_y - p_Rsn[Y]);

    // clang-format off
    gradB_p[X] += b_p[X]*dBidx[X]*Sxx + b_p[Y]*dBidx[Y]*Sxy + b_p[Z]*dBidx[Z]*Sxz;
    gradB_p[Y] += b_p[X]*dBidy[X]*Syx + b_p[Y]*dBidy[Y]*Syy + b_p[Z]*dBidy[Z]*Syz;
    gradB_p[Z] += b_p[X]*dBidz[X]*Szx + b_p[Y]*dBidz[Y]*Szy + b_p[Z]*dBidz[Z]*Szz;
    // clang-format on
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::interpolate(Vector3R& E_p, Vector3R& B_p,
  Vector3R& gradB_p, const Vector3R& rn, const Vector3R& r0)
{
  PetscFunctionBeginHot;
  E_p = {};
  B_p = {};
  gradB_p = {};

  interpolate_E(E_p, rn, r0);
  interpolate_B(B_p, rn);
  auto b_p = B_p.normalized();
  if (!gradB_g) {interpolate_gradB(gradB_p, b_p, rn, r0);}
  else {
    ::Shape shape;
    shape.setup(rn,1.0,sfunc1);
    SimpleInterpolation interpolation(shape);
    SimpleInterpolation::Context e_fields;
    SimpleInterpolation::Context gradb_fields;
    if (gradB_g != nullptr) {
      gradb_fields.emplace_back(gradB_p, gradB_g);
    }
    interpolation.process(e_fields, gradb_fields);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::decomposition_J
(const Vector3R& Rsn, const Vector3R& Rs0, const Vector3R& Vp, PetscReal q_p)
{
  PetscFunctionBeginHot;
  Vector3R p_Rsn{
    Rsn[X] / dx,
    Rsn[Y] / dy,
    Rsn[Z] / dz,
  };

  Vector3R p_Rs0{
    Rs0[X] / dx,
    Rs0[Y] / dy,
    Rs0[Z] / dz,
  };

  Vector3R p_dR{
    p_Rsn[X] != p_Rs0[X] ? p_Rsn[X] - p_Rs0[X] : 1.0,
    p_Rsn[Y] != p_Rs0[Y] ? p_Rsn[Y] - p_Rs0[Y] : 1.0,
    p_Rsn[Z] != p_Rs0[Z] ? p_Rsn[Z] - p_Rs0[Z] : 1.0,
  };

  PetscInt shr = 2;
  PetscInt shw = 2 * shr + 1;

  Vector3I p_g{
    (PetscInt)std::floor(p_Rsn[X]) - shr,
    (PetscInt)std::floor(p_Rsn[Y]) - shr,
    (PetscInt)std::floor(p_Rsn[Z]) - shr,
  };

  for (PetscInt i = 0; i < POW3(shw); ++i) {
    PetscInt g_x = p_g[X] + i % shw;
    PetscInt g_y = p_g[Y] + (i / shw) % shw;
    PetscInt g_z = p_g[Z] + (i / shw) / shw;

    Vector3R J_p;

    // clang-format off
    J_p[X] = q_p * Vp[X] * alpha * p_dR[X] * sfunc1(g_x + 0.5 - 0.5 * (p_Rsn[X] + p_Rs0[X])) *
        sfuncE(g_y - p_Rs0[Y], g_y - p_Rsn[Y], g_z - p_Rs0[Z], g_z - p_Rsn[Z]);

    J_p[Y] = q_p * Vp[Y] * alpha * p_dR[Y] * sfunc1(g_y + 0.5 - 0.5 * (p_Rsn[Y] + p_Rs0[Y])) *
        sfuncE(g_z - p_Rs0[Z], g_z - p_Rsn[Z], g_x - p_Rs0[X], g_x - p_Rsn[X]);

    J_p[Z] = q_p * Vp[Z] * alpha * p_dR[Z] * sfunc1(g_z + 0.5 - 0.5 * (p_Rsn[Z] + p_Rs0[Z])) *
        sfuncE(g_x - p_Rs0[X], g_x - p_Rsn[X], g_y - p_Rs0[Y], g_y - p_Rsn[Y]);
    // clang-format on

#pragma omp atomic
    J_g[g_z][g_y][g_x][X] += J_p[X];

#pragma omp atomic
    J_g[g_z][g_y][g_x][Y] += J_p[Y];

#pragma omp atomic
    J_g[g_z][g_y][g_x][Z] += J_p[Z];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
} 

PetscErrorCode DriftKineticEsirkepov::decomposition_M
(const Vector3R& Rsn, PetscReal mu_p)
{
  PetscFunctionBeginHot;
  ::Shape shape;
  shape.setup(Rsn, 1., sfunc1);

  Vector3R B_p = {};
  interpolate_B(B_p, Rsn);
  Vector3R b_p = B_p.normalized();

  PetscFunctionBeginHot;
  for (PetscInt i = 0; i < shape.size.elements_product(); ++i) {
    PetscInt g_x = shape.start[X] + i % shape.size[X];
    PetscInt g_y = shape.start[Y] + (i / shape.size[X]) % shape.size[Y];
    PetscInt g_z = shape.start[Z] + (i / shape.size[X]) / shape.size[Y];

    Vector3R M_shape = shape.magnetic(i);

  #pragma omp atomic update
    M_g[g_z][g_y][g_x][X] += mu_p * b_p.x() * M_shape.x();

  #pragma omp atomic update
    M_g[g_z][g_y][g_x][Y] += mu_p * b_p.y() * M_shape.y();

  #pragma omp atomic update
    M_g[g_z][g_y][g_x][Z] += mu_p * b_p.z() * M_shape.z();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
} 
