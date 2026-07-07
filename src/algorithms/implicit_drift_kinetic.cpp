#include "implicit_drift_kinetic.h"
#include "src/impls/drift_kinetic/segments.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace drift_kinetic {

constexpr auto& sfunc0 = spline_of_0th_order;

constexpr auto& sfunc1 = spline_of_1st_order;

constexpr auto& sfunc2 = spline_of_2nd_order;

PetscReal sfunc11(PetscReal left_s, PetscReal right_s)
{
  return sfunc1(left_s) * sfunc1(right_s);
}

PetscReal sfunc22(PetscReal left_s, PetscReal right_s)
{
  return sfunc2(left_s) * sfunc2(right_s);
}

const PetscReal alpha = 1.0 / 6.0;

PetscReal alongS1(PetscReal s)
{
  return sfunc1(s);
}

PetscReal alongS0(PetscReal s)
{
  return sfunc0(s);
}

PetscReal acrossS2(PetscReal left_s0, PetscReal left_sn, PetscReal right_s0, PetscReal right_sn) {
  return alpha * (sfunc22(left_s0, right_sn) + sfunc22(left_sn, right_s0) +
  2 * (sfunc22(left_s0, right_s0) + sfunc22(left_sn, right_sn)));
}

PetscReal acrossS1(PetscReal left_s0, PetscReal left_sn, PetscReal right_s0, PetscReal right_sn)
{
  return alpha * (sfunc11(left_s0, right_sn) + sfunc11(left_sn, right_s0) +
  2 * (sfunc11(left_s0, right_s0) + sfunc11(left_sn, right_sn)));
}

DriftKineticEsirkepov::DriftKineticEsirkepov(Vector3R*** B_g)
  : Bn_g(B_g) {}

DriftKineticEsirkepov::DriftKineticEsirkepov(Vector3R*** E_g, Vector3R*** Bn_g, Vector3R*** Bnh_g, Vector3R*** Bn1_g, Vector3R*** J_g, Vector3R*** M_g)
  : E_g(E_g), Bn_g(Bn_g), Bnh_g(Bnh_g), Bn1_g(Bn1_g), J_g(J_g), M_g(M_g) {}

void DriftKineticShapeGradB::setup(const DriftKineticSegment& segment){
  p_s = Vector3R{
    std::floor(segment.Rsmid[X] + 0.5),
    std::floor(segment.Rsmid[Y] + 0.5),
    std::floor(segment.Rsmid[Z] + 0.5),
  };

  p_g = Vector3I{
    (PetscInt)std::floor(segment.Rsmid[X] + 0.5) - shr,
    (PetscInt)std::floor(segment.Rsmid[Y] + 0.5) - shr,
    (PetscInt)std::floor(segment.Rsmid[Z] + 0.5) - shr,
  };
}

void DriftKineticShapeE::setup(const DriftKineticSegment& segment){
  p_s = Vector3R{
    std::floor(segment.Rsmid[X] + 0.5),
    std::floor(segment.Rsmid[Y] + 0.5),
    std::floor(segment.Rsmid[Z] + 0.5),
  };

  p_g = Vector3I{
    (PetscInt)std::floor(segment.Rsmid[X] + 0.5) - shr,
    (PetscInt)std::floor(segment.Rsmid[Y] + 0.5) - shr,
    (PetscInt)std::floor(segment.Rsmid[Z] + 0.5) - shr,
  };
}

PetscErrorCode DriftKineticEsirkepov::interpolate_B(
  Vector3R& B_p, Vector3R*** B_g, const Vector3R& Rp)
{
  PetscFunctionBeginHot;
  B_p = {};
  Shape sh_B;
  sh_B.setup(Rp, 1.5, sfunc2);

  SimpleInterpolation interpolation(sh_B);
  SimpleInterpolation::Context E_dummy;
  SimpleInterpolation::Context B_fields{{B_p, B_g}};
  interpolation.process(E_dummy, B_fields);
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode DriftKineticEsirkepov::interpolate_B(
  Vector3R& B_p, const Vector3R& Rp)
{
  PetscFunctionBeginHot;
  if (Bn1_g == nullptr) {
    interpolate_B(B_p, Bn_g, Rp);
  } else {
    interpolate_B(B_p, Bn1_g, Rp);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::interpolate_E(
  Vector3R& E_p, const DriftKineticSegment& s)
{
  PetscFunctionBeginHot;
  DriftKineticShapeE sh;
  sh.setup(s);

  for (PetscInt i = 0; i < POW3(sh.shw); ++i) {
    PetscInt g_x = sh.p_g[X] + i % sh.shw;
    PetscInt g_y = sh.p_g[Y] + (i / sh.shw) % sh.shw;
    PetscInt g_z = sh.p_g[Z] + (i / sh.shw) / sh.shw;

    auto& E = E_g[g_z][g_y][g_x];

    E_p[X] += E[X] * alongS1(g_x + 0.5 - s.Rsmid[X]) *
        acrossS2(g_y - s.Rs0[Y], g_y - s.Rsn[Y], g_z - s.Rs0[Z], g_z - s.Rsn[Z]);

    E_p[Y] += E[Y] * alongS1(g_y + 0.5 - s.Rsmid[Y]) *
        acrossS2(g_z - s.Rs0[Z], g_z - s.Rsn[Z], g_x - s.Rs0[X], g_x - s.Rsn[X]);

    E_p[Z] += E[Z] * alongS1(g_z + 0.5 - s.Rsmid[Z]) *
        acrossS2(g_x - s.Rs0[X], g_x - s.Rsn[X], g_y - s.Rs0[Y], g_y - s.Rsn[Y]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::interpolate_gradB(
  Vector3R& gradB_p, const Vector3R& b_p, const DriftKineticSegment& s)
{
  PetscFunctionBeginHot;
  DriftKineticShapeGradB sh;
  sh.setup(s);

  Vector3R dBdx, dBdy, dBdz;

  PetscReal a;
  PetscReal alongSx, alongSy, alongSz;
  PetscReal shift_ii = 0.5;
  PetscReal shift_ij = 1.;

  for (PetscInt i = 0; i < POW3(sh.shw); ++i) {
    PetscInt g_x = sh.p_g[X] + i % sh.shw;
    PetscInt g_y = sh.p_g[Y] + (i / sh.shw) % sh.shw;
    PetscInt g_z = sh.p_g[Z] + (i / sh.shw) / sh.shw;

    const Vector3R B = Bnh_g[g_z][g_y][g_x];

    dBdx = (Bnh_g[g_z][g_y][g_x + 1] - B) / dx;
    dBdy = (Bnh_g[g_z][g_y + 1][g_x] - B) / dy;
    dBdz = (Bnh_g[g_z + 1][g_y][g_x] - B) / dz;


    // X - COMPONENT

    alongSx = alongS1(g_x + shift_ii - s.Rsmid[X]);

    PetscReal Sxx = alongSx * acrossS2(g_y + 0.5 - s.Rs0[Y], g_y + 0.5 - s.Rsn[Y], //
                                       g_z + 0.5 - s.Rs0[Z], g_z + 0.5 - s.Rsn[Z]);

    a = (sh.p_s[X] - s.Rs0[X])/s.dRs[X];
    if (a > 0. && a < 1.) {
      alongSx = a * alongS1(g_x + shift_ij - 0.5 * (s.Rs0[X] + sh.p_s[X])) + (1. - a) * alongS1(g_x + shift_ij - 0.5 * (s.Rsn[X] + sh.p_s[X]));
    } else {
      alongSx = alongS1(g_x + shift_ij - s.Rsmid[X]);
    }

    PetscReal Sxy = alongSx * acrossS2(g_y - s.Rs0[Y], g_y - s.Rsn[Y], //
                                       g_z + 0.5 - s.Rs0[Z], g_z + 0.5 - s.Rsn[Z]);
    PetscReal Sxz = alongSx * acrossS2(g_y + 0.5 - s.Rs0[Y], g_y + 0.5 - s.Rsn[Y], //
                                       g_z - s.Rs0[Z], g_z - s.Rsn[Z]);

    gradB_p[X] +=  //
      b_p[X] * dBdx[X] * Sxx +  //
      b_p[Y] * dBdx[Y] * Sxy +  //
      b_p[Z] * dBdx[Z] * Sxz;

    // Y - COMPONENT

    alongSy = alongS1(g_y + shift_ii - s.Rsmid[Y]);

    PetscReal Syy = alongSy * acrossS2(g_z + 0.5 - s.Rs0[Z], g_z + 0.5 - s.Rsn[Z],
                                       g_x + 0.5 - s.Rs0[X], g_x + 0.5 - s.Rsn[X]);

    a = (sh.p_s[Y] - s.Rs0[Y])/s.dRs[Y];
    if (a > 0. && a < 1.) {
      alongSy = a * alongS1(g_y + shift_ij - 0.5 * (s.Rs0[Y] + sh.p_s[Y])) + (1. - a) * alongS1(g_y + shift_ij - 0.5 * (s.Rsn[Y] + sh.p_s[Y]));
    } else {
      alongSy = alongS1(g_y + shift_ij - s.Rsmid[Y]);
    }

    PetscReal Syx = alongSy * acrossS2(g_z + 0.5 - s.Rs0[Z], g_z + 0.5 - s.Rsn[Z],
                                       g_x - s.Rs0[X], g_x - s.Rsn[X]);
    PetscReal Syz = alongSy * acrossS2(g_z - s.Rs0[Z], g_z - s.Rsn[Z],
                                       g_x + 0.5 - s.Rs0[X], g_x + 0.5 - s.Rsn[X]);

    gradB_p[Y] +=  //
      b_p[X] * dBdy[X] * Syx +  //
      b_p[Y] * dBdy[Y] * Syy +  //
      b_p[Z] * dBdy[Z] * Syz;

    // Z - COMPONENT

    alongSz = alongS1(g_z + shift_ii - s.Rsmid[Z]);

    PetscReal Szz = alongSz * acrossS2(g_x + 0.5 - s.Rs0[X], g_x + 0.5 - s.Rsn[X],
                                       g_y + 0.5 - s.Rs0[Y], g_y + 0.5 - s.Rsn[Y]);

    a = (sh.p_s[Z] - s.Rs0[Z])/s.dRs[Z];
    if (a > 0. && a < 1.) {
      alongSz = a * alongS1(g_z + shift_ij - 0.5 * (s.Rs0[Z] + sh.p_s[Z])) + (1. - a) * alongS1(g_z + shift_ij - 0.5 * (s.Rsn[Z] + sh.p_s[Z]));
    } else {
      alongSz = alongS1(g_z + shift_ij - s.Rsmid[Z]);
    }

    PetscReal Szx = alongSz * acrossS2(g_x - s.Rs0[X], g_x - s.Rsn[X],
                                       g_y + 0.5 - s.Rs0[Y], g_y + 0.5 - s.Rsn[Y]);
    PetscReal Szy = alongSz * acrossS2(g_x + 0.5 - s.Rs0[X], g_x + 0.5 - s.Rsn[X],
                                       g_y - s.Rs0[Y], g_y - s.Rsn[Y]);

    gradB_p[Z] +=  //
      b_p[X] * dBdz[X] * Szx +  //
      b_p[Y] * dBdz[Y] * Szy +  //
      b_p[Z] * dBdz[Z] * Szz;

  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode DriftKineticEsirkepov::interpolate_rotB(
  Vector3R& rotB_p, const DriftKineticSegment& s)
{
  PetscFunctionBeginHot;
  DriftKineticShapeGradB sh;
  sh.setup(s);

  Vector3R dBdx, dBdy, dBdz;

  PetscReal a;
  PetscReal alongSx, alongSy, alongSz;
  PetscReal shift_ij = 1.;

  for (PetscInt i = 0; i < POW3(sh.shw); ++i) {
    PetscInt g_x = sh.p_g[X] + i % sh.shw;
    PetscInt g_y = sh.p_g[Y] + (i / sh.shw) % sh.shw;
    PetscInt g_z = sh.p_g[Z] + (i / sh.shw) / sh.shw;

    const Vector3R B = Bnh_g[g_z][g_y][g_x];

    dBdx = (Bnh_g[g_z][g_y][g_x + 1] - B) / dx;
    dBdy = (Bnh_g[g_z][g_y + 1][g_x] - B) / dy;
    dBdz = (Bnh_g[g_z + 1][g_y][g_x] - B) / dz;


    // X - COMPONENT

    a = (sh.p_s[X] - s.Rs0[X])/s.dRs[X];
    if (a > 0. && a < 1.) {
      alongSx = a * alongS1(g_x + shift_ij - 0.5 * (s.Rs0[X] + sh.p_s[X])) + //
                (1. - a) * alongS1(g_x + shift_ij - 0.5 * (s.Rsn[X] + sh.p_s[X]));
    } else {
      alongSx = alongS1(g_x + shift_ij - s.Rsmid[X]);
    }

    PetscReal Sxy = alongSx * acrossS2(g_y - s.Rs0[Y], g_y - s.Rsn[Y], //
                                       g_z + 0.5 - s.Rs0[Z], g_z + 0.5 - s.Rsn[Z]);
    PetscReal Sxz = alongSx * acrossS2(g_y + 0.5 - s.Rs0[Y], g_y + 0.5 - s.Rsn[Y], //
                                       g_z - s.Rs0[Z], g_z - s.Rsn[Z]);

    // Y - COMPONENT

    a = (sh.p_s[Y] - s.Rs0[Y])/s.dRs[Y];
    if (a > 0. && a < 1.) {
      alongSy = a * alongS1(g_y + shift_ij - 0.5 * (s.Rs0[Y] + sh.p_s[Y])) + //
                (1. - a) * alongS1(g_y + shift_ij - 0.5 * (s.Rsn[Y] + sh.p_s[Y]));
    } else {
      alongSy = alongS1(g_y + shift_ij - s.Rsmid[Y]);
    }

    PetscReal Syx = alongSy * acrossS2(g_z + 0.5 - s.Rs0[Z], g_z + 0.5 - s.Rsn[Z], //
                                       g_x - s.Rs0[X], g_x - s.Rsn[X]);
    PetscReal Syz = alongSy * acrossS2(g_z - s.Rs0[Z], g_z - s.Rsn[Z], //
                                       g_x + 0.5 - s.Rs0[X], g_x + 0.5 - s.Rsn[X]);

    // Z - COMPONENT

    a = (sh.p_s[Z] - s.Rs0[Z])/s.dRs[Z];
    if (a > 0. && a < 1.) {
      alongSz = a * alongS1(g_z + shift_ij - 0.5 * (s.Rs0[Z] + sh.p_s[Z])) + //
                (1. - a) * alongS1(g_z + shift_ij - 0.5 * (s.Rsn[Z] + sh.p_s[Z]));
    } else {
      alongSz = alongS1(g_z + shift_ij - s.Rsmid[Z]);
    }

    PetscReal Szx = alongSz * acrossS2(g_x - s.Rs0[X], g_x - s.Rsn[X], //
                                       g_y + 0.5 - s.Rs0[Y], g_y + 0.5 - s.Rsn[Y]);
    PetscReal Szy = alongSz * acrossS2(g_x + 0.5 - s.Rs0[X], g_x + 0.5 - s.Rsn[X], //
                                       g_y - s.Rs0[Y], g_y - s.Rsn[Y]);

    rotB_p[X] +=  //
      dBdy[Z] * Syz -  //
      dBdz[Y] * Szy;

    rotB_p[Y] +=  //
      dBdz[X] * Szx -  //
      dBdx[Z] * Sxz;

    rotB_p[Z] +=  //
      dBdx[Y] * Sxy -  //
      dBdy[X] * Syx;

  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::interpolate( //
  Vector3R& E_p, PetscReal& lenB_p, Vector3R& b_p, Vector3R& gradB_p, //
  Vector3R& rotB_p, const Vector3R& Rn, const Vector3R& R0)
{
  PetscFunctionBeginHot;

  lenB_p = 0;
  E_p = {};
  gradB_p = {};
  rotB_p = {};

  Vector3R Bn_p = {};
  Vector3R Bn1_p = {};

  DriftKineticSegment track = make_track(R0, Rn);
  std::vector<DriftKineticSegment> periodic_track = //
    periodic_segments(track, CellSplitMode::cell_centers);

  Vector3R pR0 = make_begin(periodic_track.front());
  Vector3R pRn = make_end(periodic_track.back());

  PetscCall(interpolate_B(Bn_p, Bn_g, pR0));
  PetscCall(interpolate_B(Bn1_p, Bn1_g, pRn));

  lenB_p = 0.5 * (Bn1_p.length() + Bn_p.length());

  Vector3R bn_p = Bn_p.normalized();
  Vector3R bn1_p = Bn1_p.normalized();

  b_p = 0.5 * (bn_p + bn1_p);

  PetscReal lenb_p = b_p.length();
  Vector3R hatb_p =  b_p / (lenb_p*lenb_p);

  std::vector<DriftKineticSegment> segmented_track = //
    cell_segments(periodic_track, CellSplitMode::cell_centers);

  for (const auto& segment : segmented_track) {
    Vector3R gradBs_p = {};
    Vector3R Es_p = {};
    Vector3R rotBs_p = {};
    PetscCall(interpolate_E(Es_p, segment));
    PetscCall(interpolate_gradB(gradBs_p, hatb_p, segment));
    PetscCall(interpolate_rotB(rotBs_p, segment));
    PetscReal dts = track.dRs_len > 0 ? segment.dRs_len / track.dRs_len : 1.0;
    E_p += Es_p * dts;
    gradB_p += gradBs_p * dts;
    rotB_p += rotBs_p * dts;
  }

  b_p = b_p.normalized();

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::decomposition(
  const Vector3R& Rn, const Vector3R& R0, const Vector3R& Vp, PetscReal q_p, PetscReal mu_p)
{
  PetscFunctionBeginHot;

  Vector3R Bn_p = {}, Bn1_p = {};

  DriftKineticSegment track = make_track(R0, Rn);
  std::vector<DriftKineticSegment> periodic_track =
    periodic_segments(track, CellSplitMode::cell_centers);

  Vector3R pR0 = make_begin(periodic_track.front());
  Vector3R pRn = make_end(periodic_track.back());

  PetscCall(interpolate_B(Bn_p, Bn_g, pR0));
  PetscCall(interpolate_B(Bn1_p, Bn1_g, pRn));

  Vector3R bn_p = Bn_p.normalized();
  Vector3R bn1_p = Bn1_p.normalized();
  Vector3R b_p = (bn1_p + bn_p) / 2.;

  PetscReal lenb_p = b_p.length();
  Vector3R hatb_p =  b_p / (lenb_p*lenb_p);

  PetscCall(decomposition_M(pR0, hatb_p, 0.5 * mu_p));
  PetscCall(decomposition_M(pRn, hatb_p, 0.5 * mu_p));

  std::vector<DriftKineticSegment> segmented_track =
    cell_segments(periodic_track, CellSplitMode::cell_centers);

  for (const auto& segment : segmented_track) {
    const PetscReal dts = track.dRs_len > 0 ? segment.dRs_len / track.dRs_len : 1.0;
    PetscCall(decomposition_J(segment, Vp, q_p * dts));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::decomposition_J(
  const DriftKineticSegment& s, const Vector3R& Vp, PetscReal q_p)
{
  PetscFunctionBeginHot;
  DriftKineticShapeE sh;
  sh.setup(s);
  for (PetscInt i = 0; i < POW3(sh.shw); ++i) {
    PetscInt g_x = sh.p_g[X] + i % sh.shw;
    PetscInt g_y = sh.p_g[Y] + (i / sh.shw) % sh.shw;
    PetscInt g_z = sh.p_g[Z] + (i / sh.shw) / sh.shw;

    Vector3R J_p;
    // clang-format off
    J_p[X] = q_p * Vp[X] * alongS1(g_x + 0.5 - s.Rsmid[X]) *
        acrossS2(g_y - s.Rs0[Y], g_y - s.Rsn[Y], g_z - s.Rs0[Z], g_z - s.Rsn[Z]);

    J_p[Y] = q_p * Vp[Y] * alongS1(g_y + 0.5 - s.Rsmid[Y]) *
        acrossS2(g_z - s.Rs0[Z], g_z - s.Rsn[Z], g_x - s.Rs0[X], g_x - s.Rsn[X]);

    J_p[Z] = q_p * Vp[Z] * alongS1(g_z + 0.5 - s.Rsmid[Z]) *
        acrossS2(g_x - s.Rs0[X], g_x - s.Rsn[X], g_y - s.Rs0[Y], g_y - s.Rsn[Y]);

#pragma omp atomic
    J_g[g_z][g_y][g_x][X] += J_p[X];

#pragma omp atomic
    J_g[g_z][g_y][g_x][Y] += J_p[Y];

#pragma omp atomic
    J_g[g_z][g_y][g_x][Z] += J_p[Z];
    // clang-format on
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::decomposition_M(
  const Vector3R& Rp, const Vector3R& b_p, PetscReal mu_p)
{
  Shape sh_B;
  sh_B.setup(Rp, 1.5, sfunc2);

  PetscFunctionBeginHot;
  for (PetscInt i = 0; i < sh_B.size.elements_product(); ++i) {
    PetscInt g_x = sh_B.start[X] + i % sh_B.size[X];
    PetscInt g_y = sh_B.start[Y] + (i / sh_B.size[X]) % sh_B.size[Y];
    PetscInt g_z = sh_B.start[Z] + (i / sh_B.size[X]) / sh_B.size[Y];

    Vector3R M_shape = sh_B.magnetic(i);

#pragma omp atomic
    M_g[g_z][g_y][g_x][X] -= mu_p * b_p.x() * M_shape.x();

#pragma omp atomic
    M_g[g_z][g_y][g_x][Y] -= mu_p * b_p.y() * M_shape.y();

#pragma omp atomic
    M_g[g_z][g_y][g_x][Z] -= mu_p * b_p.z() * M_shape.z();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace drift_kinetic
