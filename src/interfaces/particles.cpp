#include "particles.h"

namespace interfaces {

#define COMMUNICATION 0

#if COMMUNICATION
namespace {

constexpr PetscInt dim = 3;

PetscInt to_contiguous_index(PetscInt z, PetscInt y, PetscInt x)
{
  return indexing::petsc_index(z, y, x, 0, dim, dim, dim, 1);
}

Vector3I from_contiguous_index(PetscInt index)
{
  return Vector3I{
    (index) % dim,
    (index / dim) % dim,
    (index / dim) / dim,
  };
}

PetscInt get_index(const Vector3R& r, Axis axis, const World& world)
{
  if (r[axis] < world.start[axis])
    return 0;
  if (r[axis] < world.end[axis])
    return 1;
  return 2;
}

}  // namespace
#endif


Particles::Particles(const World& world, const SortParameters& parameters)
  : world(world), parameters(parameters), storage(geom_nz * geom_ny * geom_nx)
{
}

PetscErrorCode Particles::add_particle(const Point& point)
{
  PetscFunctionBeginUser;
  auto x = static_cast<PetscInt>(point.x() / dx);
  auto y = static_cast<PetscInt>(point.y() / dy);
  auto z = static_cast<PetscInt>(point.z() / dz);
#pragma omp critical
  storage[indexing::s_g(z, y, x)].emplace_back(point);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @todo geometry from `world` should be reused for MPI
PetscErrorCode Particles::update_cells()
{
  PetscFunctionBeginUser;
  for (PetscInt g = 0; g < geom_nz * geom_ny * geom_nx; ++g) {
    auto it = storage[g].begin();
    while (it != storage[g].end()) {
      auto ng = indexing::s_g(               //
        static_cast<PetscInt>(it->x() / dx), //
        static_cast<PetscInt>(it->y() / dy), //
        static_cast<PetscInt>(it->z() / dz));

      if (ng == g) {
        it = std::next(it);
        continue;
      }

      storage[ng].emplace_back(*it);
      it = storage[g].erase(it);
    }
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

#if COMMUNICATION
PetscErrorCode Particles::communicate()
{
  PetscFunctionBeginUser;
  constexpr PetscInt neighbors_num = POW3(3);
  std::vector<Point> outgoing[neighbors_num];
  std::vector<Point> incoming[neighbors_num];

  PetscInt center_index = to_contiguous_index(1, 1, 1);

  auto end = storage.end();
  for (auto it = storage.begin(); it != end; ++it) {
    const Vector3R& r = it->r;

    PetscInt index = to_contiguous_index( //
      get_index(r, Z, world),             //
      get_index(r, Y, world),             //
      get_index(r, X, world));

    // Particle didn't cross local boundaries
    if (index == center_index)
      continue;

    correct_coordinates(*it);

    outgoing[index].emplace_back(*it);
    std::swap(*it, *(end - 1));
    --it;
    --end;
  }
  storage.erase(end, storage.end());

  size_t o_num[neighbors_num];
  size_t i_num[neighbors_num];
  for (PetscInt i = 0; i < neighbors_num; ++i) {
    o_num[i] = outgoing[i].size();
    i_num[i] = 0;
  }

  MPI_Request reqs[2 * (neighbors_num - 1)];
  PetscInt req = 0;

  /// @note `PETSC_DEFAULT` is identical to `MPI_PROC_NULL`,
  /// so we can safely send/recv to/from neighbors.
  for (PetscInt s = 0; s < neighbors_num; ++s) {
    if (s == center_index)
      continue;

    PetscInt r = (neighbors_num - 1) - s;
    PetscCallMPI(MPI_Isend(&o_num[s], sizeof(size_t), MPI_BYTE, world.neighbors[s], MPI_TAG_NUMBERS, PETSC_COMM_WORLD, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(&i_num[r], sizeof(size_t), MPI_BYTE, world.neighbors[r], MPI_TAG_NUMBERS, PETSC_COMM_WORLD, &reqs[req++]));
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));

  req = 0;
  for (PetscInt s = 0; s < neighbors_num; ++s) {
    if (s == center_index)
      continue;

    PetscInt r = (neighbors_num - 1) - s;
    incoming[r].resize(i_num[r]);
    PetscCallMPI(MPI_Isend(outgoing[s].data(), o_num[s] * sizeof(Point), MPI_BYTE, world.neighbors[s], MPI_TAG_POINTS, PETSC_COMM_WORLD, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(incoming[r].data(), i_num[r] * sizeof(Point), MPI_BYTE, world.neighbors[r], MPI_TAG_POINTS, PETSC_COMM_WORLD, &reqs[req++]));

    if (o_num[s] > 0)
      LOG("Sending {} particles to rank {}", o_num[s], world.neighbors[s]);
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));

  for (PetscInt i = 0; i < neighbors_num; ++i) {
    if (i == center_index || i_num[i] == 0)
      continue;

    storage.insert(storage.end(),                    //
      std::make_move_iterator(incoming[i].begin()),  //
      std::make_move_iterator(incoming[i].end()));

    LOG("Receiving {} particles from rank {}", i_num[i], world.neighbors[i]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif


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
  return p / sqrt(m * m + p.squared());
}

void Particles::correct_coordinates(Point& point)
{
  if (world.bounds[X] == DM_BOUNDARY_PERIODIC)
    g_bound_periodic(point, X);
  if (world.bounds[Y] == DM_BOUNDARY_PERIODIC)
    g_bound_periodic(point, Y);
  if (world.bounds[Z] == DM_BOUNDARY_PERIODIC)
    g_bound_periodic(point, Z);
}

}  // namespace interfaces
