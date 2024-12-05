#include "particles.h"

namespace interfaces {

Particles::Particles(const World& world, const SortParameters& parameters)
  : world_(world), parameters_(parameters)
{
}

PetscErrorCode Particles::add_particle(const Point& point)
{
  PetscFunctionBeginUser;
#pragma omp critical
  points_.emplace_back(point);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::communicate()
{
  PetscFunctionBeginUser;
  constexpr PetscInt neighbors_num = POW3(3);
  std::vector<Point> outgoing[neighbors_num];
  std::vector<Point> incoming[neighbors_num];

  // clang-format off
  auto set_index = [&](const Vector3R& r, Vector3I& index, Axis axis) {
    index[axis] = (r[axis] < world_.start[axis]) ? 0 : (r[axis] < world_.end[axis]) ? 1 : 2;
  };
  // clang-format on

  auto correct_coordinates = [&](Point& point) {
    if (world_.bounds[X] == DM_BOUNDARY_PERIODIC)
      g_bound_periodic(point, X);
    if (world_.bounds[Y] == DM_BOUNDARY_PERIODIC)
      g_bound_periodic(point, Y);
    if (world_.bounds[Z] == DM_BOUNDARY_PERIODIC)
      g_bound_periodic(point, Z);
  };

  PetscInt center_index = to_contiguous_index(1, 1, 1);

  auto end = points_.end();
  for (auto it = points_.begin(); it != end; ++it) {
    const Vector3R& r = it->r;
    Vector3I v_index;
    set_index(r, v_index, X);
    set_index(r, v_index, Y);
    set_index(r, v_index, Z);

    PetscInt index = to_contiguous_index(v_index[Z], v_index[Y], v_index[X]);

    // Particle didn't cross local boundaries
    if (index == center_index)
      continue;

    correct_coordinates(*it);
    outgoing[index].emplace_back(std::move(*it));
    std::swap(*it, *(end - 1));
    --it;
    --end;
  }
  points_.erase(end, points_.end());

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
    PetscCallMPI(MPI_Isend(&o_num[s], sizeof(size_t), MPI_BYTE, world_.neighbors[s], MPI_TAG_NUMBERS, PETSC_COMM_WORLD, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(&i_num[r], sizeof(size_t), MPI_BYTE, world_.neighbors[r], MPI_TAG_NUMBERS, PETSC_COMM_WORLD, &reqs[req++]));
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));
  PetscCheck(o_num[center_index] == 0, PETSC_COMM_WORLD, PETSC_ERR_MPI, "Particles should not be passed to itself process, %ld particles was passed", o_num[center_index]);
  PetscCheck(i_num[center_index] == 0, PETSC_COMM_WORLD, PETSC_ERR_MPI, "Particles should not be passed to itself process, %ld particles was passed", i_num[center_index]);

  req = 0;
  for (PetscInt s = 0; s < neighbors_num; ++s) {
    if (s == center_index)
      continue;
    PetscInt r = (neighbors_num - 1) - s;
    incoming[r].resize(i_num[r]);
    PetscCallMPI(MPI_Isend(outgoing[s].data(), o_num[s] * sizeof(Point), MPI_BYTE, world_.neighbors[s], MPI_TAG_POINTS, PETSC_COMM_WORLD, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(incoming[r].data(), i_num[r] * sizeof(Point), MPI_BYTE, world_.neighbors[r], MPI_TAG_POINTS, PETSC_COMM_WORLD, &reqs[req++]));
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));

  for (PetscInt i = 0; i < neighbors_num; ++i) {
    if (i == center_index)
      continue;
    points_.insert(points_.end(),                    //
      std::make_move_iterator(incoming[i].begin()),  //
      std::make_move_iterator(incoming[i].end()));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


const SortParameters& Particles::parameters() const
{
  return parameters_;
}

const std::vector<Point>& Particles::points() const
{
  return points_;
}

PetscInt Particles::particles_number(const Point& point) const
{
  return parameters_.Np;
}

PetscReal Particles::density(const Point& point) const
{
  return parameters_.n;
}

PetscReal Particles::charge(const Point& point) const
{
  return parameters_.q;
}

PetscReal Particles::mass(const Point& point) const
{
  return parameters_.m;
}

Vector3R Particles::velocity(const Point& point) const
{
  const Vector3R& p = point.p;
  PetscReal m = mass(point);
  return p / sqrt(m * m + p.squared());
}

}  // namespace interfaces
