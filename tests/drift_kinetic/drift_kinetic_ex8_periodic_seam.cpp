#include "src/algorithms/implicit_drift_kinetic.h"
#include "src/utils/operators.h"
#include "src/utils/shape.h"
#include "src/utils/world.h"

using namespace drift_kinetic;

struct Track {
  Vector3R begin, end;
  PetscInt axis;
  const char* name;
};

// Check the actual ranges used by each kernel before dereferencing the DMDA arrays.
static bool within(const World& world, const Vector3I& start, const Vector3I& size,
  const Track& track, const char* stencil, PetscMPIInt rank)
{
  bool valid = true;
  for (PetscInt a = 0; a < 3; ++a) {
    if (start[a] < world.gstart[a] || start[a] + size[a] > world.gend[a]) {
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,
        "rank %d track %s axis %d: %s index range [%d,%d) outside ghost [%d,%d) on axis %d\n",
        rank, track.name, track.axis, stencil, start[a], start[a] + size[a],
        world.gstart[a], world.gend[a], a);
      valid = false;
    }
  }
  return valid;
}

static bool safe_track(const World& world, const Track& track, PetscMPIInt rank)
{
  const auto line = make_track(track.begin, track.end);
  const auto segments = cell_segments(line, CellSplitMode::centers);
  bool valid = true;
  for (const auto& segment : segments) {
    DriftKineticShapeE e;
    DriftKineticShapeGradB b;
    e.setup(segment);
    b.setup(segment);
    valid &= within(world, e.p_g, Vector3I(e.shw), track, "E/J", rank);
    // rot(B) reads one extra node in each positive direction.
    valid &= within(world, b.p_g, Vector3I(b.shw + 1), track, "gradB/rotB", rank);
  }
  for (const Vector3R& p : {make_begin(segments.front()), make_end(segments.back())}) {
    Shape shape;
    shape.setup(p, 1.5, spline_of_2nd_order);
    valid &= within(world, shape.start, shape.size, track, "endpoint B/M", rank);
  }
  return valid;
}

static PetscErrorCode deposit_charge(DM da, Vec local, const Vector3R& point, PetscReal sign)
{
  PetscFunctionBeginUser;
  PetscReal*** rho;
  PetscCall(DMDAVecGetArray(da, local, &rho));
  const Vector3R r = Shape::make_r(point);
  const Vector3I start{(PetscInt)std::ceil(r[X] - 1.5),
    (PetscInt)std::ceil(r[Y] - 1.5), (PetscInt)std::ceil(r[Z] - 1.5)};
  for (PetscInt z = 0; z < 3; ++z)
    for (PetscInt y = 0; y < 3; ++y)
      for (PetscInt x = 0; x < 3; ++x) {
        const PetscInt i = start[X] + x, j = start[Y] + y, k = start[Z] + z;
        rho[k][j][i] += sign * spline_of_2nd_order(r[X] - i) *
          spline_of_2nd_order(r[Y] - j) * spline_of_2nd_order(r[Z] - k);
      }
  PetscCall(DMDAVecRestoreArray(da, local, &rho));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode run()
{
  PetscFunctionBeginUser;
  World::set_geometry(16.0, 16.0, 16.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
  World world;
  std::ranges::fill(world.bounds, DM_BOUNDARY_PERIODIC);
  PetscCall(world.initialize());
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  std::vector<Track> tracks;
  for (PetscInt a = 0; a < 3; ++a) {
    for (PetscInt side = 0; side < 2; ++side) {
      for (PetscInt direction = 0; direction < 2; ++direction) {
        Vector3R p{8.25, 8.25, 8.25}, q = p;
        // Both seam sides and both directions; coordinates remain unwrapped.
        p[a] = side == 0 ? 0.15 : 15.85;
        q[a] = p[a] + (direction == 0 ? -0.3 : 0.3);
        tracks.push_back({p, q, a, side == 0 ? (direction ? "lower-in" : "lower-out") : (direction ? "upper-out" : "upper-in")});
      }
    }
    Vector3R p{8.25, 8.25, 8.25}, q = p;
    q[a] += 0.3;
    tracks.push_back({p, q, a, "interior"});
  }

  Vec E, B, J, M, rho, Jglobal, rhoglobal, residual;
  PetscCall(DMCreateLocalVector(world.da, &E));
  PetscCall(VecDuplicate(E, &B));
  PetscCall(VecDuplicate(E, &J));
  PetscCall(VecDuplicate(E, &M));
  PetscCall(DMCreateLocalVector(world.da_rho, &rho));
  PetscCall(DMCreateGlobalVector(world.da, &Jglobal));
  PetscCall(DMCreateGlobalVector(world.da_rho, &rhoglobal));
  PetscCall(VecDuplicate(rhoglobal, &residual));
  Mat div;
  Divergence divergence(world.da);
  PetscCall(divergence.create_negative(&div));
  PetscCall(VecSet(E, 0));
  PetscCall(VecSet(B, 0));
  Vector3R*** b;
  PetscCall(DMDAVecGetArray(world.da, B, &b));
  for (PetscInt z = world.gstart[Z]; z < world.gend[Z]; ++z)
    for (PetscInt y = world.gstart[Y]; y < world.gend[Y]; ++y)
      for (PetscInt x = world.gstart[X]; x < world.gend[X]; ++x)
        b[z][y][x] = Vector3R{0, 0, 1};
  PetscCall(DMDAVecRestoreArray(world.da, B, &b));

  PetscInt failures = 0;
  for (const auto& track : tracks) {
    bool owner = true;
    for (PetscInt a = 0; a < 3; ++a) {
      const PetscInt cell = (PetscInt)std::floor(track.begin[a] / Dx[a]);
      owner &= world.start[a] <= cell && cell < world.end[a];
    }
    const bool safe = !owner || safe_track(world, track, rank);
    PetscInt local_bad = !safe, any_bad;
    PetscCallMPI(MPI_Allreduce(&local_bad, &any_bad, 1, MPIU_INT, MPI_MAX, PETSC_COMM_WORLD));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
    if (any_bad) {
      ++failures;
      continue;
    }
    PetscCall(VecSet(J, 0));
    PetscCall(VecSet(M, 0));
    PetscCall(VecSet(rho, 0));
    PetscCall(VecSet(Jglobal, 0));
    PetscCall(VecSet(rhoglobal, 0));
    if (owner) {
      Vector3R ***e, ***bf, ***j, ***m;
      PetscCall(DMDAVecGetArray(world.da, E, &e));
      PetscCall(DMDAVecGetArray(world.da, B, &bf));
      PetscCall(DMDAVecGetArray(world.da, J, &j));
      PetscCall(DMDAVecGetArray(world.da, M, &m));
      DriftKineticEsirkepov kernel(e, bf, bf, bf, j, m);
      Vector3R ep, bhat, grad, rot;
      PetscReal len;
      PetscCall(kernel.interpolate(ep, len, bhat, grad, rot, track.end, track.begin));
      PetscCall(kernel.decomposition(track.end, track.begin, (track.end - track.begin) / dt, 1.0, 0.1));
      PetscCall(DMDAVecRestoreArray(world.da, M, &m));
      PetscCall(DMDAVecRestoreArray(world.da, J, &j));
      PetscCall(DMDAVecRestoreArray(world.da, B, &bf));
      PetscCall(DMDAVecRestoreArray(world.da, E, &e));
      PetscCall(deposit_charge(world.da_rho, rho, track.begin, -1 / dt));
      PetscCall(deposit_charge(world.da_rho, rho, track.end, 1 / dt));
    }
    PetscCall(DMLocalToGlobal(world.da, J, ADD_VALUES, Jglobal));
    PetscCall(DMLocalToGlobal(world.da_rho, rho, ADD_VALUES, rhoglobal));
    PetscCall(MatMultAdd(div, Jglobal, rhoglobal, residual));
    PetscReal norm;
    PetscCall(VecNorm(residual, NORM_2, &norm));
    if (norm >= 1e-10 || !std::isfinite(norm)) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "track %s axis %d: continuity residual %.16e\n", track.name, track.axis, (double)norm));
      ++failures;
    }
  }
  PetscCall(MatDestroy(&div));
  for (Vec* v : {&E, &B, &J, &M, &rho, &Jglobal, &rhoglobal, &residual})
    PetscCall(VecDestroy(v));
  PetscCall(DMDestroy(&world.da_rho));
  PetscCall(DMDestroy(&world.da_mflux));
  PetscCall(world.finalize());
  PetscCheck(failures == 0, PETSC_COMM_WORLD, PETSC_ERR_USER, "%d periodic seam cases failed", failures);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char** argv)
{
  PetscCall(PetscInitialize(&argc, &argv, nullptr, "DK periodic seam stencil and continuity regression.\n"));
  PetscCall(run());
  PetscCall(PetscFinalize());
  return 0;
}
