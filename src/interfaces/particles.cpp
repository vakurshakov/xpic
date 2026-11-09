#include "particles.h"

#include "src/utils/geometries.h"
#include "src/utils/profiler.h"
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
    parameters(with_macro_particle_weight(parameters)),
    storage(world)
{
}

PetscErrorCode Particles::initialize()
{
  PetscFunctionBeginUser;
  PetscCall(Profiler::register_event("update_cells", &event_update_cells));
  PetscCall(Profiler::register_event("calculate_rho", &event_calculate_rho));

  PetscCall(DMCreateGlobalVector(da, &J));
  PetscCall(DMCreateGlobalVector(da_rho, &rho[0]));
  PetscCall(DMCreateGlobalVector(da_rho, &rho[1]));

  PetscCall(DMCreateLocalVector(da, &J_loc));
  PetscCall(DMCreateLocalVector(da_rho, &rho_loc[0]));
  PetscCall(DMCreateLocalVector(da_rho, &rho_loc[1]));
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

  ParticleStorage::Attributes f;
  PetscCall(storage.get_attributes(f));
  const PointArrays arrays(f);
  for (PetscInt i = 0; i < n; i++)
    arrays.store(first + i, points[i]);
  PetscCall(storage.restore_attributes(f));

  PetscCall(storage.sort());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::extract(
  const std::function<bool(const Point&)>& predicate,
  std::vector<Point>* extracted)
{
  PetscFunctionBeginUser;
  ParticleStorage::Attributes f;
  PetscCall(storage.get_attributes(f));
  const PointArrays arrays(f);

  std::vector<PetscInt> indices;
  for (PetscInt i = 0; i < f.size; i++) {
    const Point point = arrays.load(i);
    if (!predicate(point))
      continue;
    extracted->push_back(point);
    indices.push_back(i);
  }
  PetscCall(storage.restore_attributes(f));

  PetscCall(storage.remove(std::move(indices)));
  PetscCall(storage.sort());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::update_cells()
{
  PetscFunctionBeginUser;
  PetscCall(Profiler::push_event(event_update_cells));
  PetscCall(storage.update_cells());
  PetscCall(storage.sort());
  PetscCall(Profiler::pop_event(event_update_cells));

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

  ParticleStorage::Attributes f;
  PetscCall(storage.get_attribute(f, "p"));
  const PetscReal* p = f["p"];

  for (PetscInt i = 0; i < f.size; i++) {
    for (PetscInt a : {X, Y, Z})
      hist[a][(PetscInt)(p[3 * i + a] * hist_h) + hist_h]++;
  }
  PetscCall(storage.restore_attributes(f));

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


PetscErrorCode Particles::calculate_rho()
{
  PetscFunctionBeginUser;
  PetscCall(Profiler::push_event(event_calculate_rho));

  auto r = rho[1];
  auto r_loc = rho_loc[1];
  auto r_arr = rho_arr[1];

  PetscCall(VecSwap(rho[0], r));
  PetscCall(VecSwap(rho_loc[0], r_loc));

  PetscCall(VecSet(r, 0));
  PetscCall(VecSet(r_loc, 0));

  PetscCall(DMDAVecGetArrayWrite(da_rho, r_loc, &r_arr));

  const PetscReal q = parameters.q;
  const PetscReal mpw = parameters.mpw;

  ParticleStorage::Attributes f;
  PetscCall(storage.get_attributes(f));
  const PointArrays arrays(f);

#pragma omp parallel for
  for (PetscInt n = 0; n < f.size; n++) {
    const Point point = arrays.load(n);

    for_each_shape_node(point.r,
      [&](PetscInt x, PetscInt y, PetscInt z, PetscReal w)
      {
#pragma omp atomic update
        r_arr[z][y][x] += (q * mpw) * w;
      });
  }
  PetscCall(storage.restore_attributes(f));
  PetscCall(DMDAVecRestoreArrayWrite(da_rho, r_loc, &r_arr));

  PetscCall(DMLocalToGlobal(da_rho, r_loc, ADD_VALUES, r));

  PetscCall(Profiler::pop_event(event_calculate_rho));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace interfaces
