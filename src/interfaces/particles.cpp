#include "particles.h"

#include "src/utils/geometries.h"
#include "src/utils/shape.h"

namespace interfaces {

Particles::Particles(const World& world, const SortParameters& parameters)
  : world(world),
    da(world.da),
    da_rho(world.da_rho),
    parameters(parameters),
    storage(world.size.elements_product())
{
  PetscMPIInt size;
  PetscCallAbort(PETSC_COMM_WORLD, MPI_Comm_size(PETSC_COMM_WORLD, &size));
  update_cells = (size == 1) //
    ? std::bind(std::mem_fn(&Particles::update_cells_seq), this)
    : std::bind(std::mem_fn(&Particles::update_cells_mpi), this);

  PetscCallAbort(PETSC_COMM_WORLD, PetscClassIdRegister("interfaces::Particles", &classid));
  PetscCallAbort(PETSC_COMM_WORLD, PetscLogEventRegister("update_cells", classid, &events[0]));

  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da, &J));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da_rho, &rho[0]));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da_rho, &rho[1]));

  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da, &J_loc));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da_rho, &rho_loc[0]));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da_rho, &rho_loc[1]));
}

PetscErrorCode Particles::add_particle(const Point& point, bool* is_added)
{
  PetscFunctionBeginUser;
  Vector3I vg{
    FLOOR_STEP(point.x(), dx) - world.start[X],
    FLOOR_STEP(point.y(), dy) - world.start[Y],
    FLOOR_STEP(point.z(), dz) - world.start[Z],
  };

  if (!is_point_within_bounds(vg, 0, world.size))
    PetscFunctionReturn(PETSC_SUCCESS);

#pragma omp critical
  storage[world.s_g(REP3_A(vg))].emplace_back(point);

  if (is_added) {
#pragma omp atomic write
    *is_added = true;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::correct_coordinates()
{
  PetscFunctionBeginUser;
#pragma omp parallel for
  for (auto& cell : storage)
    for (auto& point : cell)
      correct_coordinates(point);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::update_cells_seq()
{
  PetscFunctionBeginUser;
  PetscLogEventBegin(events[0], 0, 0, 0, 0);

  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    auto it = storage[g].begin();
    while (it != storage[g].end()) {
      PetscCall(correct_coordinates(*it));

      Vector3I vng{
        FLOOR_STEP(it->x(), dx),
        FLOOR_STEP(it->y(), dy),
        FLOOR_STEP(it->z(), dz),
      };

      auto ng = world.s_g(REP3_A(vng));
      if (ng == g) {
        it = std::next(it);
        continue;
      }

      if (is_point_within_bounds(vng, world.start, world.size))
        storage[ng].emplace_back(std::move(*it));

      it = storage[g].erase(it);
    }
  }

  PetscLogEventEnd(events[0], 0, 0, 0, 0);

  PetscInt sum = 0;
  for (const auto& cell : storage)
    sum += cell.size();

  LOG("  Cells have been updated, total number of particles: {}", sum);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @note The one more reason to use `DMSWARM` is the crash I encounter here with complex MPI setup
PetscErrorCode Particles::update_cells_mpi()
{
  constexpr PetscInt dim = 3;

  auto neighbor_index = [&](PetscInt x, PetscInt y, PetscInt z) {
    return indexing::petsc_index(x, y, z, 0, dim, dim, dim, 1);
  };

  auto get_index = [&](const Vector3I& r, Axis axis) {
    if (r[axis] < world.start[axis])
      return 0;
    if (r[axis] < world.end[axis])
      return 1;
    return 2;
  };

  auto get_neighbor = [&](PetscInt i) {
    return world.neighbors[i];
  };

  PetscFunctionBeginUser;
  constexpr PetscInt neighbors_num = POW3(3);
  constexpr PetscInt center_index = neighbor_index(1, 1, 1);

  std::vector<Point> outgoing[neighbors_num];
  std::vector<Point> incoming[neighbors_num];

  PetscLogEventBegin(events[0], 0, 0, 0, 0);

  LOG("  Starting MPI cells update for \"{}\"", parameters.sort_name);
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    Vector3I pg{
      world.start[X] + g % world.size[X],
      world.start[Y] + (g / world.size[X]) % world.size[Y],
      world.start[Z] + (g / world.size[X]) / world.size[Y],
    };

    auto it = storage[g].begin();
    while (it != storage[g].end()) {
      Vector3I ng{
        FLOOR_STEP(it->x(), dx),
        FLOOR_STEP(it->y(), dy),
        FLOOR_STEP(it->z(), dz),
      };

      // Particle didn't leave the cell
      if (pg[X] == ng[X] && pg[Y] == ng[Y] && pg[Z] == ng[Z]) {
        it = std::next(it);
        continue;
      }

      PetscInt i = neighbor_index(  //
        get_index(ng, X),  //
        get_index(ng, Y),  //
        get_index(ng, Z));

      // Particle didn't cross boundaries, local update cell is needed
      if (i == center_index) {
        PetscInt j = world.s_g(  //
          ng[X] - world.start[X],  //
          ng[Y] - world.start[Y],  //
          ng[Z] - world.start[Z]);

        storage[j].emplace_back(std::move(*it));
        it = storage[g].erase(it);
        continue;
      }

      // Here, neighbor-wise exchange is needed, cells will be determined after the communication
      PetscCall(correct_coordinates(*it));

      outgoing[i].emplace_back(std::move(*it));
      it = storage[g].erase(it);
    }
  }

  size_t o_num[neighbors_num];
  size_t i_num[neighbors_num];
  for (PetscInt i = 0; i < neighbors_num; ++i) {
    o_num[i] = outgoing[i].size();
    i_num[i] = 0;
  }

  MPI_Comm comm = PETSC_COMM_WORLD;

  PetscInt req = 0;
  MPI_Request reqs[2 * (neighbors_num - 1)];

  for (PetscInt s = 0; s < neighbors_num; ++s) {
    if (s == center_index)
      continue;

    PetscCallMPI(MPI_Isend(&o_num[s], 1, MPIU_SIZE_T, get_neighbor(s), MPI_TAG_NUMBERS, comm, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(&i_num[s], 1, MPIU_SIZE_T, get_neighbor(s), MPI_TAG_NUMBERS, comm, &reqs[req++]));
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));

  req = 0;
  for (PetscInt s = 0; s < neighbors_num; ++s) {
    if (s == center_index)
      continue;

    incoming[s].resize(i_num[s]);
    PetscCallMPI(MPI_Isend(outgoing[s].data(), o_num[s] * sizeof(Point), MPI_BYTE, get_neighbor(s), MPI_TAG_POINTS, comm, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(incoming[s].data(), i_num[s] * sizeof(Point), MPI_BYTE, get_neighbor(s), MPI_TAG_POINTS, comm, &reqs[req++]));
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));

  for (PetscInt i = 0; i < neighbors_num; ++i) {
    if (i == center_index || i_num[i] == 0)
      continue;

    for (auto&& point : incoming[i]) {
      PetscInt g = world.s_g(  //
        FLOOR_STEP(point.x(), dx) - world.start[X],  //
        FLOOR_STEP(point.y(), dy) - world.start[Y],  //
        FLOOR_STEP(point.z(), dz) - world.start[Z]);

      storage[g].emplace_back(std::move(point));
    }
  }

  PetscLogEventEnd(events[0], 0, 0, 0, 0);

  // Statistics of the transferred particles
  const std::vector<std::pair<std::string, size_t*>> map{
    {"    sent particles ", o_num},
    {"    received particles ", i_num},
  };

  for (auto&& [op, num] : map) {
    PetscInt sum = 0;

    for (PetscInt i = 0; i < neighbors_num; ++i)
      sum += num[i];

    PetscCall(MPIUtils::log_statistics(op, sum, comm));
  }

  PetscInt sum = 0;
  for (const auto& cell : storage)
    sum += cell.size();

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPIU_INT, MPI_SUM, comm));
  LOG("  Cells have been updated, total number of particles: {}", sum);
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscInt Particles::particles_number(const Point& /* point */) const
{
  return parameters.Np;
}

PetscReal Particles::density(const Point& /* point */) const
{
  return parameters.n;
}

PetscReal Particles::charge(const Point& /* point */) const
{
  return parameters.q;
}

PetscReal Particles::mass(const Point& /* point */) const
{
  return parameters.m;
}

PetscReal Particles::q_m(const Point& point) const
{
  return charge(point) / mass(point);
}

PetscReal Particles::n_Np(const Point& point) const
{
  return density(point) / particles_number(point);
}

PetscReal Particles::qn_Np(const Point& point) const
{
  return charge(point) * density(point) / particles_number(point);
}


/// @todo our schemes are mostly non-relativistic, we should avoid it's direct usage
Vector3R Particles::velocity(const Point& point) const
{
  const Vector3R& p = point.p;
  PetscReal m = mass(point);
  return p / std::sqrt(m * m + p.squared());
}

PetscErrorCode Particles::log_distribution() const
{
  PetscFunctionBeginUser;
  static constexpr PetscInt hist_w = 1 << 7;
  static constexpr PetscInt hist_h = hist_w / 2;
  PetscInt hist[3][hist_w];

  std::fill_n(hist[X], hist_w, 0);
  std::fill_n(hist[Y], hist_w, 0);
  std::fill_n(hist[Z], hist_w, 0);

  for (const auto& cell : storage) {
    for (const auto& point : cell) {
      hist[X][(PetscInt)(point.p[X] * hist_h) + hist_h]++;
      hist[Y][(PetscInt)(point.p[Y] * hist_h) + hist_h]++;
      hist[Z][(PetscInt)(point.p[Z] * hist_h) + hist_h]++;
    }
  }

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


PetscErrorCode Particles::correct_coordinates(Point& point)
{
  PetscFunctionBeginUser;
  if (world.bounds[X] == DM_BOUNDARY_PERIODIC)
    g_bound_periodic(point, X);
  if (world.bounds[Y] == DM_BOUNDARY_PERIODIC)
    g_bound_periodic(point, Y);
  if (world.bounds[Z] == DM_BOUNDARY_PERIODIC)
    g_bound_periodic(point, Z);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::finalize()
{
  PetscFunctionBeginUser;
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

PetscErrorCode Particles::initialize_rho()
{
  PetscFunctionBeginUser;
  PetscCall(calculate_rho());
  PetscCall(VecCopy(rho[1], rho[0]));
  PetscCall(VecCopy(rho_loc[1], rho_loc[0]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::calculate_rho()
{
  PetscFunctionBeginUser;
  auto r = rho[1];
  auto r_loc = rho_loc[1];
  auto r_arr = rho_arr[1];

  PetscCall(VecSwap(rho[0], r));
  PetscCall(VecSwap(rho_loc[0], r_loc));

  PetscCall(VecSet(r, 0));
  PetscCall(VecSet(r_loc, 0));

  PetscCall(DMDAVecGetArrayWrite(da_rho, r_loc, &r_arr));

  DefaultRhoShape shape;
  const PetscReal q = parameters.q;
  const PetscReal mpw = parameters.n / (PetscReal)parameters.Np;

#pragma omp parallel for private(shape)
  for (auto&& cell : storage) {
    for (auto&& point : cell) {
      shape.setup(point.r);

      PetscInt i, x, y, z;

      for (i = 0; i < shape.shm; ++i) {
        x = shape.start[X] + i % shape.shw;
        y = shape.start[Y] + (i / shape.shw) % shape.shw;
        z = shape.start[Z] + (i / shape.shw) / shape.shw;

#pragma omp atomic update
        r_arr[z][y][x] += (q * mpw) * shape.cache[i];
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da_rho, r_loc, &r_arr));
  PetscCall(DMLocalToGlobal(da_rho, r_loc, ADD_VALUES, r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace interfaces
