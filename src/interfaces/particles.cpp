#include "particles.h"

#include "src/utils/geometries.h"
#include "src/utils/shape.h"

namespace interfaces {

static SortParameters with_macro_particle_weight(SortParameters parameters)
{
  parameters.mpw = parameters.n / (PetscReal)parameters.Np;
  return parameters;
}

Particles::Particles(const World& world, const SortParameters& parameters)
  : world(world),
    da(world.da),
    da_rho(world.da_rho),
    da_mflux(world.da_mflux),
    parameters(with_macro_particle_weight(parameters)),
    storage(world)
{
}

PetscErrorCode Particles::initialize()
{
  PetscFunctionBeginUser;
  PetscCall(PetscClassIdRegister("interfaces::Particles", &classid));
  PetscCall(PetscLogEventRegister("update_cells", classid, &events[0]));

  PetscCall(DMCreateGlobalVector(da, &J));
  PetscCall(DMCreateGlobalVector(da_rho, &rho[0]));
  PetscCall(DMCreateGlobalVector(da_rho, &rho[1]));
  PetscCall(DMCreateGlobalVector(da_mflux, &mflux));

  PetscCall(DMCreateLocalVector(da, &J_loc));
  PetscCall(DMCreateLocalVector(da_rho, &rho_loc[0]));
  PetscCall(DMCreateLocalVector(da_rho, &rho_loc[1]));
  PetscCall(DMCreateLocalVector(da_mflux, &mflux_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

bool Particles::is_local(const Point& point) const
{
  Vector3I vg{
    FLOOR_STEP(point.x(), dx) - world.start[X],
    FLOOR_STEP(point.y(), dy) - world.start[Y],
    FLOOR_STEP(point.z(), dz) - world.start[Z],
  };
  return is_point_within_bounds(vg, 0, world.size);
}

PetscErrorCode Particles::ingest(const Point* points, PetscInt n)
{
  PetscFunctionBeginUser;
  PetscInt first;
  PetscCall(storage.grow(n, &first));

  ParticleStorage::Fields f;
  PetscCall(storage.get_fields(f));
  const PointArrays arrays(f);
  for (PetscInt i = 0; i < n; i++)
    arrays.store(first + i, points[i]);
  PetscCall(storage.restore_fields(f));

  PetscCall(storage.sort());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::extract(
  const std::function<bool(const Point&)>& predicate,
  std::vector<Point>* extracted)
{
  PetscFunctionBeginUser;
  ParticleStorage::Fields f;
  PetscCall(storage.get_fields(f));
  const PointArrays arrays(f);

  std::vector<PetscInt> indices;
  for (PetscInt i = 0; i < f.size; i++) {
    const Point point = arrays.load(i);
    if (!predicate(point))
      continue;
    extracted->push_back(point);
    indices.push_back(i);
  }
  PetscCall(storage.restore_fields(f));

  PetscCall(storage.remove(std::move(indices)));
  PetscCall(storage.sort());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::update_cells()
{
  PetscFunctionBeginUser;
  PetscLogEventBegin(events[0], 0, 0, 0, 0);
  PetscCall(storage.update_cells());
  PetscCall(storage.sort());
  PetscLogEventEnd(events[0], 0, 0, 0, 0);

  PetscInt n;
  PetscCall(storage.get_size(&n));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  LOG("  Cells have been updated, total number of particles: {}", n);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @todo our schemes are mostly non-relativistic, we should avoid it's direct usage
Vector3R Particles::velocity(const Point& point) const
{
  const Vector3R& p = point.p;
  const PetscReal m = parameters.m;
  return p / std::sqrt(m * m + p.squared());
}

PetscErrorCode Particles::log_distribution() const
{
  PetscFunctionBeginUser;
  static constexpr PetscInt hist_w = 1 << 7;
  static constexpr PetscInt hist_h = hist_w / 2;
  PetscInt hist[3][hist_w];

  std::ranges::fill(hist[X], 0);
  std::ranges::fill(hist[Y], 0);
  std::ranges::fill(hist[Z], 0);

  ParticleStorage::Fields f;
  PetscCall(storage.get_field(f, "p"));
  const PetscReal* p = f["p"];

  for (PetscInt i = 0; i < f.size; i++) {
    for (PetscInt a : {X, Y, Z})
      hist[a][(PetscInt)(p[3 * i + a] * hist_h) + hist_h]++;
  }
  PetscCall(storage.restore_fields(f));

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, hist[X], hist_w, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, hist[Y], hist_w, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, hist[Z], hist_w, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

  LOG("    {} velocity distribution histogram:", parameters.sort_name);
  LOG("    -------------------------------------------------");
  LOG("    {:3s},  {:>6s}:  {:>9s}  {:>9s}  {:>9s}", "bin", "v[c]", "hist[X]", "hist[Y]", "hist[Z]");
  for (PetscInt i = 0; i < hist_w; i++) {
    LOG("    {:3d},  {: 5.3f}:  {:9d}  {:9d}  {:9d}", i,  i / (PetscReal)hist_h - 1, hist[X][i], hist[Y][i], hist[Z][i]);
  }
  LOG("    -------------------------------------------------");
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(storage.finalize());
  PetscCall(VecDestroy(&J));
  PetscCall(VecDestroy(&rho[0]));
  PetscCall(VecDestroy(&rho[1]));

  PetscCall(VecDestroy(&J_loc));
  PetscCall(VecDestroy(&rho_loc[0]));
  PetscCall(VecDestroy(&rho_loc[1]));
  PetscFunctionReturn(PETSC_SUCCESS);
}


struct DefaultRhoShape {
  DefaultRhoShape() = default;

  static constexpr PetscReal shr = shape_radius;
  static constexpr PetscInt shw = (PetscInt)(2 * shr);
  static constexpr PetscInt shm = POW3(shw);
  static constexpr PetscReal (&sfunc)(PetscReal) = shape_function;

  Vector3I start;
  PetscReal cache[shm];

  void setup(const Vector3R& r)
  {
    Vector3R p_r = ::Shape::make_r(r);

    start = Vector3I{
      (PetscInt)(ceil(p_r[X] - shr)),
      (PetscInt)(ceil(p_r[Y] - shr)),
      (PetscInt)(ceil(p_r[Z] - shr)),
    };

#pragma omp simd
    for (PetscInt i = 0; i < shm; ++i) {
      auto x = (PetscReal)(start[X] + i % shw);
      auto y = (PetscReal)(start[Y] + (i / shw) % shw);
      auto z = (PetscReal)(start[Z] + (i / shw) / shw);
      cache[i] = sfunc(p_r[X] - x) * sfunc(p_r[Y] - y) * sfunc(p_r[Z] - z);
    }
  }
};

PetscErrorCode Particles::calculate_moments()
{
  PetscFunctionBeginUser;
  auto r = rho[1];
  auto r_loc = rho_loc[1];
  auto r_arr = rho_arr[1];

  PetscCall(VecSwap(rho[0], r));
  PetscCall(VecSwap(rho_loc[0], r_loc));

  PetscCall(VecSet(r, 0));
  PetscCall(VecSet(r_loc, 0));
  PetscCall(VecSet(mflux, 0));
  PetscCall(VecSet(mflux_loc, 0));

  PetscCall(DMDAVecGetArrayWrite(da_rho, r_loc, &r_arr));
  PetscCall(DMDAVecGetArrayDOFWrite(da_mflux, mflux_loc, &mflux_arr));

  DefaultRhoShape shape;
  const PetscReal q = parameters.q;
  const PetscReal m = parameters.m;
  const PetscReal mpw = parameters.mpw;

  ParticleStorage::Fields f;
  PetscCall(storage.get_fields(f));
  const PointArrays arrays(f);

#pragma omp parallel for private(shape)
  for (PetscInt n = 0; n < f.size; n++) {
    const Point point = arrays.load(n);
    const Vector3R& v = point.p;

    shape.setup(point.r);

    constexpr PetscInt msize = 6;
    const PetscReal moments[msize] = {
      m * v[X] * v[X],
      m * v[X] * v[Y],
      m * v[X] * v[Z],
      m * v[Y] * v[Y],
      m * v[Y] * v[Z],
      m * v[Z] * v[Z],
    };

    PetscInt i, j, x, y, z;

    for (i = 0; i < shape.shm; ++i) {
      x = shape.start[X] + i % shape.shw;
      y = shape.start[Y] + (i / shape.shw) % shape.shw;
      z = shape.start[Z] + (i / shape.shw) / shape.shw;

#pragma omp atomic update
      r_arr[z][y][x] += (q * mpw) * shape.cache[i];

      for (j = 0; j < msize; ++j) {
#pragma omp atomic update
        mflux_arr[z][y][x][j] += (moments[j] * mpw) * shape.cache[i];
      }
    }
  }
  PetscCall(storage.restore_fields(f));

  PetscCall(DMDAVecRestoreArrayWrite(da_rho, r_loc, &r_arr));
  PetscCall(DMDAVecRestoreArrayDOFWrite(da_mflux, mflux_loc, &mflux_arr));

  PetscCall(DMLocalToGlobal(da_rho, r_loc, ADD_VALUES, r));
  PetscCall(DMLocalToGlobal(da_mflux, mflux_loc, ADD_VALUES, mflux));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace interfaces
