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

}  // namespace

void DriftKineticShapeE::setup(const DriftKineticSegment& segment)
{
  p_g = segment.cell - Vector3I(shr);
}

void DriftKineticShapeGradB::setup(const DriftKineticSegment& segment)
{
  p_g = segment.cell - Vector3I(shr);
}

// S2[a][shift][end][k] is S2 at the node g_a + shift / 2 from the
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

      along_ii[a][k] = sfunc1(g + 0.5 - s.Rsh[a]);

      if (0 < frac && frac < 1) {
        along_ij[a][k] = frac * sfunc1(g + 1.0 - 0.5 * (s.Rs0[a] + p_s)) +
          (1 - frac) * sfunc1(g + 1.0 - 0.5 * (s.Rsn[a] + p_s));
      }
      else {
        along_ij[a][k] = sfunc1(g + 1.0 - s.Rsh[a]);
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

DriftKineticEsirkepov::DriftKineticEsirkepov(const World& world, Vector3R*** B_g)
  : world(world), Bn0_g(B_g)
{
}

DriftKineticEsirkepov::DriftKineticEsirkepov(const World& world,
  Vector3R*** E_g, Vector3R*** Bn0_g, Vector3R*** Bnh_g, Vector3R*** Bn1_g,
  Vector3R*** J_g, Vector3R*** M_g)
  : world(world),
    E_g(E_g),
    Bn0_g(Bn0_g),
    Bnh_g(Bnh_g),
    Bn1_g(Bn1_g),
    J_g(J_g),
    M_g(M_g)
{
}

void DriftKineticEsirkepov::set_interval(PetscReal begin, PetscReal end)
{
  theta_begin = begin;
  theta_end = end;
}

Vector3R DriftKineticEsirkepov::magnetic(
  PetscInt x, PetscInt y, PetscInt z, PetscReal theta) const
{
  if (theta == 0)
    return Bn0_g[z][y][x];
  if (theta == 1)
    return Bn1_g[z][y][x];
  if (theta == 0.5)
    return Bnh_g[z][y][x];
  return (1 - theta) * Bn0_g[z][y][x] + theta * Bn1_g[z][y][x];
}

PetscErrorCode DriftKineticEsirkepov::check_stencil(
  const Vector3I& begin, const Vector3I& end) const
{
  PetscFunctionBeginHot;
  for (Axis a : {X, Y, Z}) {
    PetscCheck(begin[a] >= world.gstart[a] && end[a] <= world.gend[a],
      PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
      "Drift-kinetic stencil outside local ghost grid on axis %d", (int)a);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::check_track(
  const Vector3R& R0, const Vector3R& Rn) const
{
  PetscFunctionBeginHot;
  for (Axis a : {X, Y, Z}) {
    for (PetscReal r : {R0[a] / Dx[a], Rn[a] / Dx[a]})
      PetscCheck(std::isfinite(r) && std::abs(r) < static_cast<PetscReal>(PETSC_MAX_INT) / 4,
        PETSC_COMM_SELF, PETSC_ERR_FP, "Invalid drift-kinetic track coordinate");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::interpolate_B_at(
  Vector3R& B_p, const Vector3R& Rp, PetscReal theta)
{
  PetscFunctionBeginHot;
  if (theta == 0 || theta == 1) {
    PetscCall(interpolate_B(B_p, theta == 0 ? Bn0_g : Bn1_g, Rp));
  }
  else {
    Shape sh;
    PetscCall(check_track(Rp, Rp));
    sh.setup(Rp, 1.5, sfunc2);
    PetscCall(check_stencil(sh.start, sh.start + sh.size));
    SimpleInterpolation interpolation(sh);
    Vector3R a, b;
    PetscCall(interpolation.process({}, {{a, Bn0_g}, {b, Bn1_g}}));
    B_p = (1 - theta) * a + theta * b;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::interpolate(                     //
  Vector3R& E_p, PetscReal& lenB_p, Vector3R& b_p, Vector3R& gradB_p,  //
  Vector3R& rotB_p, const Vector3R& Rn, const Vector3R& R0)
{
  PetscFunctionBeginHot;
  E_p = {};
  rotB_p = {};
  gradB_p = {};

  PetscCall(check_track(R0, Rn));
  DriftKineticSegment track = make_track(R0, Rn);

  std::vector<DriftKineticSegment> segmented_track =
    cell_segments(track, CellSplitMode::centers, SegmentImage::local);

  Vector3R pR0 = make_begin(segmented_track.front());
  Vector3R pRn = make_end(segmented_track.back());

  Vector3R Bn0_p, Bn1_p;
  PetscCall(interpolate_B_at(Bn0_p, pR0, theta_begin));
  PetscCall(interpolate_B_at(Bn1_p, pRn, theta_end));
  lenB_p = 0.5 * (Bn0_p.length() + Bn1_p.length());

  Vector3R bn0_p = Bn0_p.normalized();
  Vector3R bn1_p = Bn1_p.normalized();
  b_p = 0.5 * (bn0_p + bn1_p);

  Vector3R hatb_p = b_p / b_p.squared();

  for (const auto& segment : segmented_track) {
    PetscCall(check_stencil(segment.cell - Vector3I(2), segment.cell + Vector3I(3)));
    tables.setup(segment);
    Vector3R Es_p;
    Vector3R rotBs_p;
    Vector3R gradBs_p;
    PetscCall(interpolate_E(Es_p, segment));
    PetscCall(interpolate_gradB_rotB(gradBs_p, rotBs_p, hatb_p, segment));

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
  PetscCall(check_track(R0, Rn));
  DriftKineticSegment track = make_track(R0, Rn);

  std::vector<DriftKineticSegment> segmented_track =
    cell_segments(track, CellSplitMode::centers, SegmentImage::local);

  Vector3R pR0 = make_begin(segmented_track.front());
  Vector3R pRn = make_end(segmented_track.back());

  Vector3R Bn0_p, Bn1_p;
  PetscCall(interpolate_B_at(Bn0_p, pR0, theta_begin));
  PetscCall(interpolate_B_at(Bn1_p, pRn, theta_end));

  Vector3R bn0_p = Bn0_p.normalized();
  Vector3R bn1_p = Bn1_p.normalized();
  Vector3R b_p = (bn1_p + bn0_p) / 2.;

  Vector3R hatb_p = b_p / b_p.squared();

  PetscCall(decomposition_M(pR0, hatb_p, 0.5 * mu_p));
  PetscCall(decomposition_M(pRn, hatb_p, 0.5 * mu_p));

  for (const auto& segment : segmented_track) {
    PetscCall(check_stencil(segment.cell - Vector3I(2), segment.cell + Vector3I(3)));
    PetscReal dtau = track.dRs_len > 0 ? segment.dRs_len / track.dRs_len : 1.0;
    tables.setup(segment);
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
  PetscCall(check_track(Rp, Rp));
  sh.setup(Rp, 1.5, sfunc2);
  PetscCall(check_stencil(sh.start, sh.start + sh.size));

  SimpleInterpolation interpolation(sh);
  interpolation.process({}, {{B_p, B_g}});
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The E stencil is the nodes k = 1..3 of the gradB one.
PetscErrorCode DriftKineticEsirkepov::interpolate_E(
  Vector3R& E_p, const DriftKineticSegment& s)
{
  DriftKineticShapeE sh;
  sh.setup(s);
  PetscCall(check_stencil(sh.p_g, sh.p_g + Vector3I(sh.shw)));

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
  PetscCall(check_stencil(sh.p_g, sh.p_g + Vector3I(sh.shw + 1)));

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

    const Vector3R B = magnetic(g_x, g_y, g_z, 0.5 * (theta_begin + theta_end));

    const Vector3R dB[3]{
      (magnetic(g_x + 1, g_y, g_z, 0.5 * (theta_begin + theta_end)) - B) / dx,
      (magnetic(g_x, g_y + 1, g_z, 0.5 * (theta_begin + theta_end)) - B) / dy,
      (magnetic(g_x, g_y, g_z + 1, 0.5 * (theta_begin + theta_end)) - B) / dz,
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

      gradB_p[a] +=                    //
        b_p[X] * dB[a][X] * W[a][X] +  //
        b_p[Y] * dB[a][Y] * W[a][Y] +  //
        b_p[Z] * dB[a][Z] * W[a][Z];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DriftKineticEsirkepov::decomposition_M(
  const Vector3R& Rp, const Vector3R& b_p, PetscReal mu_p)
{
  Shape sh;
  PetscCall(check_track(Rp, Rp));
  sh.setup(Rp, 1.5, sfunc2);
  PetscCall(check_stencil(sh.start, sh.start + sh.size));

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

// Same weights as `interpolate_E`, as energy conservation requires.
PetscErrorCode DriftKineticEsirkepov::decomposition_J(
  const DriftKineticSegment& s, const Vector3R& Vh, PetscReal q_p)
{
  DriftKineticShapeE sh;
  sh.setup(s);
  PetscCall(check_stencil(sh.p_g, sh.p_g + Vector3I(sh.shw)));

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

}  // namespace drift_kinetic
