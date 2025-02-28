#include "particles.h"

namespace interfaces {

namespace {

constexpr PetscInt dim = 3;

constexpr PetscInt neighbor_index(PetscInt z, PetscInt y, PetscInt x)
{
  return indexing::petsc_index(z, y, x, 0, dim, dim, dim, 1);
}

constexpr PetscInt get_index(const Vector3I& r, Axis axis, const World& world)
{
  if (r[axis] < world.start[axis])
    return 0;
  if (r[axis] < world.end[axis])
    return 1;
  return 2;
}

/// @note `PETSC_DEFAULT` isn't identical to `MPI_PROC_NULL`
constexpr PetscMPIInt get_neighbor(PetscInt i, const World& world)
{
  return world.neighbors[i] < 0 ? MPI_PROC_NULL : world.neighbors[i];
}

}  // namespace


Particles::Particles(const World& world, const SortParameters& parameters)
  : world(world), parameters(parameters), storage(world.size.elements_product())
{
}

PetscErrorCode Particles::add_particle(const Point& point, bool* is_added)
{
  PetscFunctionBeginUser;
  PetscInt x = FLOOR_STEP(point.x(), dx) - world.start[X];
  PetscInt y = FLOOR_STEP(point.y(), dy) - world.start[Y];
  PetscInt z = FLOOR_STEP(point.z(), dz) - world.start[Z];

  bool within_bounds = //
    (0 <= x && x < world.size[X]) && //
    (0 <= y && y < world.size[Y]) && //
    (0 <= z && z < world.size[Z]);

  if (!within_bounds)
    PetscFunctionReturn(PETSC_SUCCESS);

#pragma omp critical
  storage[world.s_g(z, y, x)].emplace_back(point);

  if (is_added)
    *is_added = true;
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

PetscErrorCode Particles::update_cells()
{
  PetscFunctionBeginUser;
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    auto it = storage[g].begin();
    while (it != storage[g].end()) {
      auto ng = world.s_g(       //
        FLOOR_STEP(it->x(), dx), //
        FLOOR_STEP(it->y(), dy), //
        FLOOR_STEP(it->z(), dz));

      if (ng == g) {
        it = std::next(it);
        continue;
      }

      PetscCall(correct_coordinates(*it));

      storage[ng].emplace_back(std::move(*it));
      it = storage[g].erase(it);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::update_cells_mpi()
{
  PetscFunctionBeginUser;
  constexpr PetscInt neighbors_num = POW3(3);
  constexpr PetscInt center_index = neighbor_index(1, 1, 1);

  std::vector<Point> outgoing[neighbors_num];
  std::vector<Point> incoming[neighbors_num];

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

      PetscInt i = neighbor_index( //
        get_index(ng, Z, world),   //
        get_index(ng, Y, world),   //
        get_index(ng, X, world));

      // Particle didn't cross boundaries, local update cell is needed
      if (i == center_index) {
        PetscInt j = world.s_g(   //
          ng[Z] - world.start[Z], //
          ng[Y] - world.start[Y], //
          ng[X] - world.start[X]);

        storage[j].emplace_back(std::move(*it));
        it = storage[g].erase(it);
        continue;
      }

      // Here, neighbor-wise exchange is needed, cells will be determined after the communication
      correct_coordinates(*it);

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

    PetscInt r = (neighbors_num - 1) - s;
    PetscCallMPI(MPI_Isend(&o_num[s], 1, MPIU_SIZE_T, get_neighbor(s, world), MPI_TAG_NUMBERS, comm, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(&i_num[r], 1, MPIU_SIZE_T, get_neighbor(r, world), MPI_TAG_NUMBERS, comm, &reqs[req++]));
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));

  req = 0;
  for (PetscInt s = 0; s < neighbors_num; ++s) {
    if (s == center_index)
      continue;

    PetscInt r = (neighbors_num - 1) - s;
    incoming[r].resize(i_num[r]);
    PetscCallMPI(MPI_Isend(outgoing[s].data(), o_num[s] * sizeof(Point), MPI_BYTE, get_neighbor(s, world), MPI_TAG_POINTS, comm, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(incoming[r].data(), i_num[r] * sizeof(Point), MPI_BYTE, get_neighbor(r, world), MPI_TAG_POINTS, comm, &reqs[req++]));
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));

  for (PetscInt i = 0; i < neighbors_num; ++i) {
    if (i == center_index || i_num[i] == 0)
      continue;

    for (auto&& point : incoming[i]) {
      PetscInt g = world.s_g(  //
        FLOOR_STEP(point.z(), dz) - world.start[Z],  //
        FLOOR_STEP(point.y(), dy) - world.start[Y],  //
        FLOOR_STEP(point.x(), dx) - world.start[X]);

      storage[g].emplace_back(std::move(point));
    }
  }

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

Vector3R Particles::velocity(const Point& point) const
{
  const Vector3R& p = point.p;
  PetscReal m = mass(point);
  return p / std::sqrt(m * m + p.squared());
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

}  // namespace interfaces
