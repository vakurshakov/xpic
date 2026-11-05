#include "implicit_drift_kinetic.h"

#include "src/algorithms/simple_interpolation.h"

namespace drift_kinetic {

namespace {

// Copies of `spline_of_1st/2nd_order`: with internal linkage they are inlined
// into the stencil loops instead of being called through the PLT of libxpic.so.
PetscReal sfunc1(PetscReal s)
{
  s = std::abs(s);
  return (s <= 1.0) ? 1.0 - s : 0.0;
}

PetscReal sfunc2(PetscReal s)
{
  s = std::abs(s);

  if (s <= 0.5)
    return 0.75 - s * s;
  if (s < 1.5)
    return 0.5 * (1.5 - s) * (1.5 - s);
  return 0.0;
}

const PetscReal alpha = 1.0 / 6.0;

PetscReal alongS1(PetscReal s)
{
  return sfunc1(s);
}

#if !DRIFT_KINETIC_SPLINE_TABLES
PetscReal sfunc22(PetscReal left_s, PetscReal right_s)
{
  return sfunc2(left_s) * sfunc2(right_s);
}

PetscReal acrossS2(
  PetscReal left_s0, PetscReal left_sn, PetscReal right_s0, PetscReal right_sn)
{
  return alpha *
    (sfunc22(left_s0, right_sn) + sfunc22(left_sn, right_s0) +
      2 * (sfunc22(left_s0, right_s0) + sfunc22(left_sn, right_sn)));
}
#endif

}  // namespace

void DriftKineticShapeE::setup(const DriftKineticSegment& segment)
{
  p_g = segment.cell - Vector3I(shr);
}

void DriftKineticShapeGradB::setup(const DriftKineticSegment& segment)
{
  p_s = Vector3R{
    (PetscReal)segment.cell[X],
    (PetscReal)segment.cell[Y],
    (PetscReal)segment.cell[Z],
  };

  p_g = segment.cell - Vector3I(shr);
}

#if DRIFT_KINETIC_SPLINE_TABLES
// Same expressions as the per-node evaluation, so the weights are bitwise
// identical; S2[a][shift][end][k] is S2 at the node g_a + shift / 2 from the
// segment start (end = 0) or end (end = 1).
void DriftKineticSplineTables::setup(const DriftKineticSegment& s)
{
  const Vector3I p_g = s.cell - Vector3I(DriftKineticShapeGradB::shr);
  PetscReal S2[3][2][2][shw];

  for (Axis a : {X, Y, Z}) {
    const auto p_s = (PetscReal)s.cell[a];
    const PetscReal frac = (p_s - s.Rs0[a]) / s.dRs[a];

    for (PetscInt k = 0; k < shw; k++) {
      const PetscInt g = p_g[a] + k;

      S2[a][0][0][k] = sfunc2(g - s.Rs0[a]);
      S2[a][0][1][k] = sfunc2(g - s.Rsn[a]);
      S2[a][1][0][k] = sfunc2(g + 0.5 - s.Rs0[a]);
      S2[a][1][1][k] = sfunc2(g + 0.5 - s.Rsn[a]);

      along_ii[a][k] = alongS1(g + 0.5 - s.Rsh[a]);

      if (0 < frac && frac < 1) {
        along_ij[a][k] = frac * alongS1(g + 1.0 - 0.5 * (s.Rs0[a] + p_s)) +
          (1 - frac) * alongS1(g + 1.0 - 0.5 * (s.Rsn[a] + p_s));
      }
      else {
        along_ij[a][k] = alongS1(g + 1.0 - s.Rsh[a]);
      }
    }
  }

  for (PetscInt a = 0; a < 3; a++) {
    const PetscInt b = (a + 1) % 3;
    const PetscInt c = (a + 2) % 3;

    for (PetscInt m = 0; m < 4; m++) {
      const auto& Sb = S2[b][m / 2];
      const auto& Sc = S2[c][m % 2];

      for (PetscInt kb = 0; kb < shw; kb++) {
        for (PetscInt kc = 0; kc < shw; kc++) {
          across[a][m][kb][kc] = alpha *
            (Sb[0][kb] * Sc[1][kc] + Sb[1][kb] * Sc[0][kc] +
              2 * (Sb[0][kb] * Sc[0][kc] + Sb[1][kb] * Sc[1][kc]));
        }
      }
    }
  }
}
#endif

DriftKineticEsirkepov::DriftKineticEsirkepov(const World& world, Vector3R*** B_g)
  : world(world), Bn0_g(B_g)
{
}

DriftKineticEsirkepov::DriftKineticEsirkepov(const World& world, Vector3R*** E_g, Vector3R*** Bn0_g,
  Vector3R*** Bnh_g, Vector3R*** Bn1_g, Vector3R*** J_g, Vector3R*** M_g)
  : world(world), E_g(E_g), Bn0_g(Bn0_g), Bnh_g(Bnh_g), Bn1_g(Bn1_g), J_g(J_g), M_g(M_g)
{
}

void DriftKineticEsirkepov::check_stencil(
  const Vector3I& start, const Vector3I& size, const char* name) const
{
  for (PetscInt a = 0; a < 3; ++a)
    PetscAssertAbort(world.gstart[a] <= start[a] && start[a] + size[a] <= world.gend[a], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
      "DK %s stencil axis %d: [%d,%d) outside ghost range [%d,%d)", name, a, start[a], start[a] + size[a], world.gstart[a], world.gend[a]);
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

  std::vector<DriftKineticSegment> segmented_track =
    cell_segments(track, CellSplitMode::centers, SegmentImage::local);

  Vector3R pR0 = make_begin(segmented_track.front());
  Vector3R pRn = make_end(segmented_track.back());

  Vector3R Bn0_p, Bn1_p;
  PetscCall(interpolate_B(Bn0_p, Bn0_g, pR0));
  PetscCall(interpolate_B(Bn1_p, Bn1_g, pRn));
  lenB_p = 0.5 * (Bn0_p.length() + Bn1_p.length());

  Vector3R bn0_p = Bn0_p.normalized();
  Vector3R bn1_p = Bn1_p.normalized();
  b_p = 0.5 * (bn0_p + bn1_p);

  Vector3R hatb_p = b_p / b_p.squared();

  for (const auto& segment : segmented_track) {
#if DRIFT_KINETIC_SPLINE_TABLES
    tables.setup(segment);
#endif
    Vector3R Es_p;
    Vector3R rotBs_p;
    Vector3R gradBs_p;
    PetscCall(interpolate_E(Es_p, segment));
#if DRIFT_KINETIC_SPLINE_TABLES
    PetscCall(interpolate_gradB_rotB(gradBs_p, rotBs_p, hatb_p, segment));
#else
    PetscCall(interpolate_rotB(rotBs_p, segment));
    PetscCall(interpolate_gradB(gradBs_p, hatb_p, segment));
#endif

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

  std::vector<DriftKineticSegment> segmented_track =
    cell_segments(track, CellSplitMode::centers, SegmentImage::local);

  Vector3R pR0 = make_begin(segmented_track.front());
  Vector3R pRn = make_end(segmented_track.back());

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
#if DRIFT_KINETIC_SPLINE_TABLES
    tables.setup(segment);
#endif
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
  check_stencil(sh.start, sh.size, "interpolate_B");

  SimpleInterpolation interpolation(sh);
  interpolation.process({}, {{B_p, B_g}});
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if DRIFT_KINETIC_SPLINE_TABLES
// The E stencil is the nodes k = 1..3 of the gradB one.
PetscErrorCode DriftKineticEsirkepov::interpolate_E(
  Vector3R& E_p, const DriftKineticSegment& s)
{
  DriftKineticShapeE sh;
  sh.setup(s);
  check_stencil(sh.p_g, Vector3I(sh.shw), "interpolate_E");

  PetscFunctionBeginHot;
  for (PetscInt i = 0; i < POW3(sh.shw); ++i) {
    const PetscInt k[3]{
      1 + i % sh.shw,
      1 + (i / sh.shw) % sh.shw,
      1 + (i / sh.shw) / sh.shw,
    };

    PetscInt g_x = sh.p_g[X] + k[X] - 1;
    PetscInt g_y = sh.p_g[Y] + k[Y] - 1;
    PetscInt g_z = sh.p_g[Z] + k[Z] - 1;

    const auto& t = tables;
    auto& E = E_g[g_z][g_y][g_x];

    E_p[X] += E[X] * t.along_ii[X][k[X]] * t.across[X][0][k[Y]][k[Z]];
    E_p[Y] += E[Y] * t.along_ii[Y][k[Y]] * t.across[Y][0][k[Z]][k[X]];
    E_p[Z] += E[Z] * t.along_ii[Z][k[Z]] * t.across[Z][0][k[X]][k[Y]];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// One pass over the stencil for both quantities: they share the derivatives
// of B and the off-diagonal weights S_ac, collected per node into W[a][c].
PetscErrorCode DriftKineticEsirkepov::interpolate_gradB_rotB(Vector3R& gradB_p,
  Vector3R& rotB_p, const Vector3R& b_p, const DriftKineticSegment& s)
{
  DriftKineticShapeGradB sh;
  sh.setup(s);
  check_stencil(sh.p_g, Vector3I(sh.shw + 1), "interpolate_gradB_rotB");

  const auto& t = tables;

  PetscFunctionBeginHot;
  for (PetscInt i = 0; i < POW3(sh.shw); ++i) {
    const PetscInt k[3]{
      i % sh.shw,
      (i / sh.shw) % sh.shw,
      (i / sh.shw) / sh.shw,
    };

    PetscInt g_x = sh.p_g[X] + k[X];
    PetscInt g_y = sh.p_g[Y] + k[Y];
    PetscInt g_z = sh.p_g[Z] + k[Z];

    const Vector3R B = Bnh_g[g_z][g_y][g_x];

    const Vector3R dB[3]{
      (Bnh_g[g_z][g_y][g_x + 1] - B) / dx,
      (Bnh_g[g_z][g_y + 1][g_x] - B) / dy,
      (Bnh_g[g_z + 1][g_y][g_x] - B) / dz,
    };

    PetscReal W[3][3];
    for (PetscInt a = 0; a < 3; a++) {
      const PetscInt b = (a + 1) % 3;
      const PetscInt c = (a + 2) % 3;
      W[a][a] = t.along_ii[a][k[a]] * t.across[a][3][k[b]][k[c]];
      W[a][b] = t.along_ij[a][k[a]] * t.across[a][1][k[b]][k[c]];
      W[a][c] = t.along_ij[a][k[a]] * t.across[a][2][k[b]][k[c]];
    }

    for (PetscInt a = 0; a < 3; a++) {
      const PetscInt b = (a + 1) % 3;
      const PetscInt c = (a + 2) % 3;

      rotB_p[a] += dB[b][c] * W[b][c] - dB[c][b] * W[c][b];

      gradB_p[a] +=                       //
        b_p[X] * dB[a][X] * W[a][X] +  //
        b_p[Y] * dB[a][Y] * W[a][Y] +  //
        b_p[Z] * dB[a][Z] * W[a][Z];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#else
PetscErrorCode DriftKineticEsirkepov::interpolate_E(
  Vector3R& E_p, const DriftKineticSegment& s)
{
  DriftKineticShapeE sh;
  sh.setup(s);
  check_stencil(sh.p_g, Vector3I(sh.shw), "interpolate_E");

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
  check_stencil(sh.p_g, Vector3I(sh.shw + 1), "interpolate_rotB");

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
  check_stencil(sh.p_g, Vector3I(sh.shw + 1), "interpolate_gradB");

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
#endif

PetscErrorCode DriftKineticEsirkepov::decomposition_M(
  const Vector3R& Rp, const Vector3R& b_p, PetscReal mu_p)
{
  Shape sh;
  sh.setup(Rp, 1.5, sfunc2);
  check_stencil(sh.start, sh.size, "decomposition_M");

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

#if DRIFT_KINETIC_SPLINE_TABLES
// Same weights as `interpolate_E`, as energy conservation requires.
PetscErrorCode DriftKineticEsirkepov::decomposition_J(
  const DriftKineticSegment& s, const Vector3R& Vh, PetscReal q_p)
{
  DriftKineticShapeE sh;
  sh.setup(s);
  check_stencil(sh.p_g, Vector3I(sh.shw), "decomposition_J");

  PetscFunctionBeginHot;
  for (PetscInt i = 0; i < POW3(sh.shw); ++i) {
    const PetscInt k[3]{
      1 + i % sh.shw,
      1 + (i / sh.shw) % sh.shw,
      1 + (i / sh.shw) / sh.shw,
    };

    PetscInt g_x = sh.p_g[X] + k[X] - 1;
    PetscInt g_y = sh.p_g[Y] + k[Y] - 1;
    PetscInt g_z = sh.p_g[Z] + k[Z] - 1;

    const auto& t = tables;
    Vector3R J_p;

    J_p[X] = q_p * Vh[X] * t.along_ii[X][k[X]] * t.across[X][0][k[Y]][k[Z]];
    J_p[Y] = q_p * Vh[Y] * t.along_ii[Y][k[Y]] * t.across[Y][0][k[Z]][k[X]];
    J_p[Z] = q_p * Vh[Z] * t.along_ii[Z][k[Z]] * t.across[Z][0][k[X]][k[Y]];

#pragma omp atomic
    J_g[g_z][g_y][g_x][X] += J_p[X];

#pragma omp atomic
    J_g[g_z][g_y][g_x][Y] += J_p[Y];

#pragma omp atomic
    J_g[g_z][g_y][g_x][Z] += J_p[Z];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#else
PetscErrorCode DriftKineticEsirkepov::decomposition_J(
  const DriftKineticSegment& s, const Vector3R& Vh, PetscReal q_p)
{
  DriftKineticShapeE sh;
  sh.setup(s);
  check_stencil(sh.p_g, Vector3I(sh.shw), "decomposition_J");

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
#endif

}  // namespace drift_kinetic
