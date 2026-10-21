#include "implicit_drift_kinetic.h"

#include "src/algorithms/simple_interpolation.h"

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

PetscReal acrossS2(
  PetscReal left_s0, PetscReal left_sn, PetscReal right_s0, PetscReal right_sn)
{
  return alpha *
    (sfunc22(left_s0, right_sn) + sfunc22(left_sn, right_s0) +
      2 * (sfunc22(left_s0, right_s0) + sfunc22(left_sn, right_sn)));
}

PetscReal acrossS1(
  PetscReal left_s0, PetscReal left_sn, PetscReal right_s0, PetscReal right_sn)
{
  return alpha *
    (sfunc11(left_s0, right_sn) + sfunc11(left_sn, right_s0) +
      2 * (sfunc11(left_s0, right_s0) + sfunc11(left_sn, right_sn)));
}

void DriftKineticShapeE::setup(const DriftKineticSegment& segment)
{
  p_g = Vector3I{
    (PetscInt)std::floor(segment.Rsh[X] + 0.5) - shr,
    (PetscInt)std::floor(segment.Rsh[Y] + 0.5) - shr,
    (PetscInt)std::floor(segment.Rsh[Z] + 0.5) - shr,
  };
}

void DriftKineticShapeGradB::setup(const DriftKineticSegment& segment)
{
  p_s = Vector3R{
    std::floor(segment.Rsh[X] + 0.5),
    std::floor(segment.Rsh[Y] + 0.5),
    std::floor(segment.Rsh[Z] + 0.5),
  };

  p_g = Vector3I{
    (PetscInt)std::floor(segment.Rsh[X] + 0.5) - shr,
    (PetscInt)std::floor(segment.Rsh[Y] + 0.5) - shr,
    (PetscInt)std::floor(segment.Rsh[Z] + 0.5) - shr,
  };
}

DriftKineticEsirkepov::DriftKineticEsirkepov(Vector3R*** B_g)
  : Bn0_g(B_g)
{
}

DriftKineticEsirkepov::DriftKineticEsirkepov(Vector3R*** E_g, Vector3R*** Bn0_g,
  Vector3R*** Bnh_g, Vector3R*** Bn1_g, Vector3R*** J_g, Vector3R*** M_g)
  : E_g(E_g), Bn0_g(Bn0_g), Bnh_g(Bnh_g), Bn1_g(Bn1_g), J_g(J_g), M_g(M_g)
{
}

PetscErrorCode DriftKineticEsirkepov::interpolate(                     //
  Vector3R& E_p, PetscReal& lenB_p, Vector3R& b_p, Vector3R& gradB_p,  //
  Vector3R& rotB_p, const Vector3R& Rn, const Vector3R& R0)
{
  PetscFunctionBeginHot;
  E_p = {};
  rotB_p = {};
  gradB_p = {};

  DriftKineticSegment track = make_track(R0, Rn);

  std::vector<DriftKineticSegment> periodic_track =
    periodic_segments(track, CellSplitMode::centers);

  std::vector<DriftKineticSegment> segmented_track =
    cell_segments(periodic_track, CellSplitMode::centers);

  Vector3R pR0 = make_begin(periodic_track.front());
  Vector3R pRn = make_end(periodic_track.back());

  Vector3R Bn0_p, Bn1_p;
  PetscCall(interpolate_B(Bn0_p, Bn0_g, pR0));
  PetscCall(interpolate_B(Bn1_p, Bn1_g, pRn));
  lenB_p = 0.5 * (Bn0_p.length() + Bn1_p.length());

  Vector3R bn0_p = Bn0_p.normalized();
  Vector3R bn1_p = Bn1_p.normalized();
  b_p = 0.5 * (bn0_p + bn1_p);

  Vector3R hatb_p = b_p / b_p.squared();

  for (const auto& segment : segmented_track) {
    Vector3R Es_p;
    Vector3R rotBs_p;
    Vector3R gradBs_p;
    PetscCall(interpolate_E(Es_p, segment));
    PetscCall(interpolate_rotB(rotBs_p, segment));
    PetscCall(interpolate_gradB(gradBs_p, hatb_p, segment));

    PetscReal dtau = track.dRs_len > 0 ? segment.dRs_len / track.dRs_len : 1.0;
    E_p += Es_p * dtau;
    rotB_p += rotBs_p * dtau;
    gradB_p += gradBs_p * dtau;
  }

  b_p = b_p.normalized();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::decomposition(const Vector3R& Rn,
  const Vector3R& R0, const Vector3R& Vh, PetscReal q_p, PetscReal mu_p)
{
  PetscFunctionBeginHot;
  DriftKineticSegment track = make_track(R0, Rn);

  std::vector<DriftKineticSegment> periodic_track =
    periodic_segments(track, CellSplitMode::centers);

  std::vector<DriftKineticSegment> segmented_track =
    cell_segments(periodic_track, CellSplitMode::centers);

  Vector3R pR0 = make_begin(periodic_track.front());
  Vector3R pRn = make_end(periodic_track.back());

  Vector3R Bn0_p, Bn1_p;
  PetscCall(interpolate_B(Bn0_p, Bn0_g, pR0));
  PetscCall(interpolate_B(Bn1_p, Bn1_g, pRn));

  Vector3R bn0_p = Bn0_p.normalized();
  Vector3R bn1_p = Bn1_p.normalized();
  Vector3R b_p = (bn1_p + bn0_p) / 2.;

  Vector3R hatb_p = b_p / b_p.squared();

  PetscCall(decomposition_M(pR0, hatb_p, 0.5 * mu_p));
  PetscCall(decomposition_M(pRn, hatb_p, 0.5 * mu_p));

  for (const auto& segment : segmented_track) {
    PetscReal dtau = track.dRs_len > 0 ? segment.dRs_len / track.dRs_len : 1.0;
    PetscCall(decomposition_J(segment, Vh, q_p * dtau));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::interpolate_B(
  Vector3R& B_p, const Vector3R& Rp)
{
  PetscFunctionBeginHot;
  B_p = {};
  PetscCall(interpolate_B(B_p, (Bn1_g == nullptr) ? Bn0_g : Bn1_g, Rp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::interpolate_B(
  Vector3R& B_p, Vector3R*** B_g, const Vector3R& Rp)
{
  PetscFunctionBeginHot;
  Shape sh;
  sh.setup(Rp, 1.5, sfunc2);

  SimpleInterpolation interpolation(sh);
  interpolation.process({}, {{B_p, B_g}});
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::interpolate_E(
  Vector3R& E_p, const DriftKineticSegment& s)
{
  DriftKineticShapeE sh;
  sh.setup(s);

  PetscFunctionBeginHot;
  for (PetscInt i = 0; i < POW3(sh.shw); ++i) {
    PetscInt g_x = sh.p_g[X] + i % sh.shw;
    PetscInt g_y = sh.p_g[Y] + (i / sh.shw) % sh.shw;
    PetscInt g_z = sh.p_g[Z] + (i / sh.shw) / sh.shw;

    auto& E = E_g[g_z][g_y][g_x];

    E_p[X] += E[X] * alongS1(g_x + 0.5 - s.Rsh[X]) *
      acrossS2(g_y - s.Rs0[Y], g_y - s.Rsn[Y], g_z - s.Rs0[Z], g_z - s.Rsn[Z]);

    E_p[Y] += E[Y] * alongS1(g_y + 0.5 - s.Rsh[Y]) *
      acrossS2(g_z - s.Rs0[Z], g_z - s.Rsn[Z], g_x - s.Rs0[X], g_x - s.Rsn[X]);

    E_p[Z] += E[Z] * alongS1(g_z + 0.5 - s.Rsh[Z]) *
      acrossS2(g_x - s.Rs0[X], g_x - s.Rsn[X], g_y - s.Rs0[Y], g_y - s.Rsn[Y]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::interpolate_rotB(
  Vector3R& rotB_p, const DriftKineticSegment& s)
{
  DriftKineticShapeGradB sh;
  sh.setup(s);

  Vector3R dBdx, dBdy, dBdz;

  PetscReal a;
  PetscReal alongSx, alongSy, alongSz;
  PetscReal Sxy, Sxz, Syx, Syz, Szx, Szy;
  PetscReal shift_ij = 1.;

  PetscFunctionBeginHot;
  for (PetscInt i = 0; i < POW3(sh.shw); ++i) {
    PetscInt g_x = sh.p_g[X] + i % sh.shw;
    PetscInt g_y = sh.p_g[Y] + (i / sh.shw) % sh.shw;
    PetscInt g_z = sh.p_g[Z] + (i / sh.shw) / sh.shw;

    const Vector3R B = Bnh_g[g_z][g_y][g_x];

    dBdx = (Bnh_g[g_z][g_y][g_x + 1] - B) / dx;
    dBdy = (Bnh_g[g_z][g_y + 1][g_x] - B) / dy;
    dBdz = (Bnh_g[g_z + 1][g_y][g_x] - B) / dz;


    // X - COMPONENT

    if (a = (sh.p_s[X] - s.Rs0[X]) / s.dRs[X]; 0 < a && a < 1) {
      alongSx = a * alongS1(g_x + shift_ij - 0.5 * (s.Rs0[X] + sh.p_s[X])) +  //
        (1 - a) * alongS1(g_x + shift_ij - 0.5 * (s.Rsn[X] + sh.p_s[X]));
    }
    else {
      alongSx = alongS1(g_x + shift_ij - s.Rsh[X]);
    }

    Sxy = alongSx *
      acrossS2(g_y - s.Rs0[Y], g_y - s.Rsn[Y],  //
        g_z + 0.5 - s.Rs0[Z], g_z + 0.5 - s.Rsn[Z]);

    Sxz = alongSx *
      acrossS2(g_y + 0.5 - s.Rs0[Y], g_y + 0.5 - s.Rsn[Y],  //
        g_z - s.Rs0[Z], g_z - s.Rsn[Z]);

    // Y - COMPONENT

    if (a = (sh.p_s[Y] - s.Rs0[Y]) / s.dRs[Y]; 0 < a && a < 1) {
      alongSy = a * alongS1(g_y + shift_ij - 0.5 * (s.Rs0[Y] + sh.p_s[Y])) +  //
        (1 - a) * alongS1(g_y + shift_ij - 0.5 * (s.Rsn[Y] + sh.p_s[Y]));
    }
    else {
      alongSy = alongS1(g_y + shift_ij - s.Rsh[Y]);
    }

    Syx = alongSy *
      acrossS2(g_z + 0.5 - s.Rs0[Z], g_z + 0.5 - s.Rsn[Z],  //
        g_x - s.Rs0[X], g_x - s.Rsn[X]);

    Syz = alongSy *
      acrossS2(g_z - s.Rs0[Z], g_z - s.Rsn[Z],  //
        g_x + 0.5 - s.Rs0[X], g_x + 0.5 - s.Rsn[X]);

    // Z - COMPONENT

    if (a = (sh.p_s[Z] - s.Rs0[Z]) / s.dRs[Z]; 0 < a && a < 1) {
      alongSz = a * alongS1(g_z + shift_ij - 0.5 * (s.Rs0[Z] + sh.p_s[Z])) +  //
        (1 - a) * alongS1(g_z + shift_ij - 0.5 * (s.Rsn[Z] + sh.p_s[Z]));
    }
    else {
      alongSz = alongS1(g_z + shift_ij - s.Rsh[Z]);
    }

    Szx = alongSz *
      acrossS2(g_x - s.Rs0[X], g_x - s.Rsn[X],  //
        g_y + 0.5 - s.Rs0[Y], g_y + 0.5 - s.Rsn[Y]);

    Szy = alongSz *
      acrossS2(g_x + 0.5 - s.Rs0[X], g_x + 0.5 - s.Rsn[X],  //
        g_y - s.Rs0[Y], g_y - s.Rsn[Y]);

    rotB_p[X] += dBdy[Z] * Syz - dBdz[Y] * Szy;
    rotB_p[Y] += dBdz[X] * Szx - dBdx[Z] * Sxz;
    rotB_p[Z] += dBdx[Y] * Sxy - dBdy[X] * Syx;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::interpolate_gradB(
  Vector3R& gradB_p, const Vector3R& b_p, const DriftKineticSegment& s)
{
  DriftKineticShapeGradB sh;
  sh.setup(s);

  Vector3R dBdx, dBdy, dBdz;

  PetscReal a;
  PetscReal alongSx, alongSy, alongSz;
  PetscReal Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz;
  PetscReal shift_ii = 0.5;
  PetscReal shift_ij = 1.;

  PetscFunctionBeginHot;
  for (PetscInt i = 0; i < POW3(sh.shw); ++i) {
    PetscInt g_x = sh.p_g[X] + i % sh.shw;
    PetscInt g_y = sh.p_g[Y] + (i / sh.shw) % sh.shw;
    PetscInt g_z = sh.p_g[Z] + (i / sh.shw) / sh.shw;

    const Vector3R B = Bnh_g[g_z][g_y][g_x];

    dBdx = (Bnh_g[g_z][g_y][g_x + 1] - B) / dx;
    dBdy = (Bnh_g[g_z][g_y + 1][g_x] - B) / dy;
    dBdz = (Bnh_g[g_z + 1][g_y][g_x] - B) / dz;


    // X - COMPONENT

    alongSx = alongS1(g_x + shift_ii - s.Rsh[X]);

    Sxx = alongSx *
      acrossS2(g_y + 0.5 - s.Rs0[Y], g_y + 0.5 - s.Rsn[Y],  //
        g_z + 0.5 - s.Rs0[Z], g_z + 0.5 - s.Rsn[Z]);

    if (a = (sh.p_s[X] - s.Rs0[X]) / s.dRs[X]; 0 < a && a < 1) {
      alongSx = a * alongS1(g_x + shift_ij - 0.5 * (s.Rs0[X] + sh.p_s[X])) +
        (1 - a) * alongS1(g_x + shift_ij - 0.5 * (s.Rsn[X] + sh.p_s[X]));
    }
    else {
      alongSx = alongS1(g_x + shift_ij - s.Rsh[X]);
    }

    Sxy = alongSx *
      acrossS2(g_y - s.Rs0[Y], g_y - s.Rsn[Y],  //
        g_z + 0.5 - s.Rs0[Z], g_z + 0.5 - s.Rsn[Z]);

    Sxz = alongSx *
      acrossS2(g_y + 0.5 - s.Rs0[Y], g_y + 0.5 - s.Rsn[Y],  //
        g_z - s.Rs0[Z], g_z - s.Rsn[Z]);

    gradB_p[X] +=               //
      b_p[X] * dBdx[X] * Sxx +  //
      b_p[Y] * dBdx[Y] * Sxy +  //
      b_p[Z] * dBdx[Z] * Sxz;


    // Y - COMPONENT

    alongSy = alongS1(g_y + shift_ii - s.Rsh[Y]);

    Syy = alongSy *
      acrossS2(g_z + 0.5 - s.Rs0[Z], g_z + 0.5 - s.Rsn[Z],  //
        g_x + 0.5 - s.Rs0[X], g_x + 0.5 - s.Rsn[X]);

    if (a = (sh.p_s[Y] - s.Rs0[Y]) / s.dRs[Y]; 0 < a && a < 1) {
      alongSy = a * alongS1(g_y + shift_ij - 0.5 * (s.Rs0[Y] + sh.p_s[Y])) +
        (1 - a) * alongS1(g_y + shift_ij - 0.5 * (s.Rsn[Y] + sh.p_s[Y]));
    }
    else {
      alongSy = alongS1(g_y + shift_ij - s.Rsh[Y]);
    }

    Syx = alongSy *
      acrossS2(g_z + 0.5 - s.Rs0[Z], g_z + 0.5 - s.Rsn[Z],  //
        g_x - s.Rs0[X], g_x - s.Rsn[X]);

    Syz = alongSy *
      acrossS2(g_z - s.Rs0[Z], g_z - s.Rsn[Z],  //
        g_x + 0.5 - s.Rs0[X], g_x + 0.5 - s.Rsn[X]);

    gradB_p[Y] +=               //
      b_p[X] * dBdy[X] * Syx +  //
      b_p[Y] * dBdy[Y] * Syy +  //
      b_p[Z] * dBdy[Z] * Syz;


    // Z - COMPONENT

    alongSz = alongS1(g_z + shift_ii - s.Rsh[Z]);

    Szz = alongSz *
      acrossS2(g_x + 0.5 - s.Rs0[X], g_x + 0.5 - s.Rsn[X],  //
        g_y + 0.5 - s.Rs0[Y], g_y + 0.5 - s.Rsn[Y]);

    if (a = (sh.p_s[Z] - s.Rs0[Z]) / s.dRs[Z]; 0 < a && a < 1) {
      alongSz = a * alongS1(g_z + shift_ij - 0.5 * (s.Rs0[Z] + sh.p_s[Z])) +
        (1 - a) * alongS1(g_z + shift_ij - 0.5 * (s.Rsn[Z] + sh.p_s[Z]));
    }
    else {
      alongSz = alongS1(g_z + shift_ij - s.Rsh[Z]);
    }

    Szx = alongSz *
      acrossS2(g_x - s.Rs0[X], g_x - s.Rsn[X],  //
        g_y + 0.5 - s.Rs0[Y], g_y + 0.5 - s.Rsn[Y]);

    Szy = alongSz *
      acrossS2(g_x + 0.5 - s.Rs0[X], g_x + 0.5 - s.Rsn[X],  //
        g_y - s.Rs0[Y], g_y - s.Rsn[Y]);

    gradB_p[Z] +=               //
      b_p[X] * dBdz[X] * Szx +  //
      b_p[Y] * dBdz[Y] * Szy +  //
      b_p[Z] * dBdz[Z] * Szz;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::decomposition_M(
  const Vector3R& Rp, const Vector3R& b_p, PetscReal mu_p)
{
  Shape sh;
  sh.setup(Rp, 1.5, sfunc2);

  PetscFunctionBeginHot;
  for (PetscInt i = 0; i < sh.size.elements_product(); ++i) {
    PetscInt g_x = sh.start[X] + i % sh.size[X];
    PetscInt g_y = sh.start[Y] + (i / sh.size[X]) % sh.size[Y];
    PetscInt g_z = sh.start[Z] + (i / sh.size[X]) / sh.size[Y];

    Vector3R M_shape = sh.magnetic(i);

#pragma omp atomic
    M_g[g_z][g_y][g_x][X] -= mu_p * b_p.x() * M_shape.x();

#pragma omp atomic
    M_g[g_z][g_y][g_x][Y] -= mu_p * b_p.y() * M_shape.y();

#pragma omp atomic
    M_g[g_z][g_y][g_x][Z] -= mu_p * b_p.z() * M_shape.z();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::decomposition_J(
  const DriftKineticSegment& s, const Vector3R& Vh, PetscReal q_p)
{
  DriftKineticShapeE sh;
  sh.setup(s);

  PetscFunctionBeginHot;
  for (PetscInt i = 0; i < POW3(sh.shw); ++i) {
    PetscInt g_x = sh.p_g[X] + i % sh.shw;
    PetscInt g_y = sh.p_g[Y] + (i / sh.shw) % sh.shw;
    PetscInt g_z = sh.p_g[Z] + (i / sh.shw) / sh.shw;

    Vector3R J_p;

    J_p[X] = q_p * Vh[X] * alongS1(g_x + 0.5 - s.Rsh[X]) *
      acrossS2(g_y - s.Rs0[Y], g_y - s.Rsn[Y], g_z - s.Rs0[Z], g_z - s.Rsn[Z]);

    J_p[Y] = q_p * Vh[Y] * alongS1(g_y + 0.5 - s.Rsh[Y]) *
      acrossS2(g_z - s.Rs0[Z], g_z - s.Rsn[Z], g_x - s.Rs0[X], g_x - s.Rsn[X]);

    J_p[Z] = q_p * Vh[Z] * alongS1(g_z + 0.5 - s.Rsh[Z]) *
      acrossS2(g_x - s.Rs0[X], g_x - s.Rsn[X], g_y - s.Rs0[Y], g_y - s.Rsn[Y]);

#pragma omp atomic
    J_g[g_z][g_y][g_x][X] += J_p[X];

#pragma omp atomic
    J_g[g_z][g_y][g_x][Y] += J_p[Y];

#pragma omp atomic
    J_g[g_z][g_y][g_x][Z] += J_p[Z];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace drift_kinetic
