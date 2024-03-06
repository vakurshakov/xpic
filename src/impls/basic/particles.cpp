#include "particles.h"

#include "src/impls/basic/simulation.h"

namespace basic {

enum Shift : PetscInt {
  NO = 0,  // shape[x - i]
  SH = 1   // shape[x - (i + 0.5)]
};

struct Particles::Shape {
  // `Vector3_dim` is used as a coordinate space dimensionality
  PetscReal shape[shape_width * shape_width * shape_width * Vector3_dim * 2];

  #pragma omp declare simd linear(g: 1), notinbranch
  constexpr PetscReal& operator()(PetscInt g, PetscInt comp, PetscInt shift) {
    return shape[(g * Vector3_dim + comp) * 2 + shift];
  }
};


Particles::Particles(const Simulation& simulation, const Particles_parameters& parameters)
  : parameters_(parameters), simulation_(simulation) {
  const DM& da = simulation_.da();

  PetscCallVoid(DMDAGetNeighbors(da, &neighbours));

  Vector3<PetscInt> start, size;
  PetscCallVoid(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  l_start.x() = (PetscReal)start.x() * dx;
  l_start.y() = (PetscReal)start.y() * dy;
  l_start.z() = (PetscReal)start.z() * dz;

  l_end.x() = l_start.x() + (PetscReal)size.x() * dx;
  l_end.y() = l_start.y() + (PetscReal)size.y() * dy;
  l_end.z() = l_start.z() + (PetscReal)size.z() * dz;

  l_width.x() = std::min(shape_width, geom_nx);
  l_width.y() = std::min(shape_width, geom_ny);
  l_width.z() = std::min(shape_width, geom_nz);
}


PetscErrorCode Particles::add_particle(const Point& point) {
  PetscFunctionBeginUser;
  const Vector3<PetscReal>& r = point.r;
  if (l_start.x() <= r.x() && r.x() < l_end.x() &&
      l_start.y() <= r.y() && r.y() < l_end.y() &&
      l_start.z() <= r.z() && r.z() < l_end.z()) {
    #pragma omp critical
    points_.emplace_back(point);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::push() {
  PetscFunctionBegin;
  const DM& da = simulation_.da();
  PetscCall(DMGetLocalVector(da, &local_E));
  PetscCall(DMGetLocalVector(da, &local_B));

  /// @note May we can put `DMGlobalToLocalEnd()` into `Particles::interpolate()`?
  PetscCall(DMGlobalToLocal(da, simulation_.E(), INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, simulation_.B(), INSERT_VALUES, local_B));

  PetscCall(DMDAVecGetArrayRead(da, local_E, &E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));

  #pragma omp for schedule(monotonic: dynamic, OMP_CHUNK_SIZE)
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    // Particle's coordinate (global) in `PetscReal` units of dx, dy, dz
    const Vector3<PetscReal> p_r{
      it->r.x() / dx,
      it->r.y() / dy,
      it->r.z() / dz,
    };

    // Nearest grid point to particle. Shifted by `shape_radius`
    // to use indexing in range `[0, shape_width)` later
    const Vector3<PetscInt> p_g{
      (geom_nx > 1) ? ROUND(p_r.x()) - shape_radius : 0,
      (geom_ny > 1) ? ROUND(p_r.y()) - shape_radius : 0,
      (geom_nz > 1) ? ROUND(p_r.z()) - shape_radius : 0,
    };

    static Shape shape;
    #pragma omp threadprivate(shape)

    fill_shape(p_r, p_g, shape);

    Vector3<PetscReal> point_E = 0.0;
    Vector3<PetscReal> point_B = 0.0;
    interpolate(p_g, shape, point_E, point_B);
    push(point_E, point_B, *it);
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));

  PetscCall(DMRestoreLocalVector(da, &local_E));
  PetscCall(DMRestoreLocalVector(da, &local_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}


void Particles::fill_shape(const Vector3<PetscReal>& p_r, const Vector3<PetscInt>& p_g, Shape& shape) {
  PetscInt g_x, g_y, g_z;

  #pragma omp simd collapse(Vector3_dim)
  for (PetscInt z = 0; z < l_width[Z]; ++z) {
  for (PetscInt y = 0; y < l_width[Y]; ++y) {
  for (PetscInt x = 0; x < l_width[X]; ++x) {
    PetscInt i = ((z * shape_width + y) * shape_width + x);
    g_x = p_g[X] + x;
    g_y = p_g[Y] + y;
    g_z = p_g[Z] + z;

    shape(i, X, NO) = (geom_nx > 1) ? shape_function(p_r.x() - g_x) : 1.0;
    shape(i, Y, NO) = (geom_ny > 1) ? shape_function(p_r.y() - g_y) : 1.0;
    shape(i, Z, NO) = (geom_nz > 1) ? shape_function(p_r.z() - g_z) : 1.0;

    shape(i, X, SH) = (geom_nx > 1) ? shape_function(p_r.x() - (g_x + 0.5)) : 1.0;
    shape(i, Y, SH) = (geom_ny > 1) ? shape_function(p_r.y() - (g_y + 0.5)) : 1.0;
    shape(i, Z, SH) = (geom_nz > 1) ? shape_function(p_r.z() - (g_z + 0.5)) : 1.0;
  }}}
}


void Particles::interpolate(const Vector3<PetscInt>& p_g, Shape& shape, Vector3<PetscReal>& point_E, Vector3<PetscReal>& point_B) const {
  PetscInt g_x, g_y, g_z;

  #pragma omp simd collapse(Vector3_dim)
  for (PetscInt z = 0; z < l_width[Z]; ++z) {
  for (PetscInt y = 0; y < l_width[Y]; ++y) {
  for (PetscInt x = 0; x < l_width[X]; ++x) {
    PetscInt i = ((z * shape_width + y) * shape_width + x);
    g_x = p_g[X] + x;
    g_y = p_g[Y] + y;
    g_z = p_g[Z] + z;

    point_E.x() += E[g_z][g_y][g_x].x() * shape(i, Z, NO) * shape(i, Y, NO) * shape(i, X, SH);
    point_E.y() += E[g_z][g_y][g_x].y() * shape(i, Z, NO) * shape(i, Y, SH) * shape(i, X, NO);
    point_E.z() += E[g_z][g_y][g_x].z() * shape(i, Z, SH) * shape(i, Y, NO) * shape(i, X, NO);

    point_B.x() += B[g_z][g_y][g_x].x() * shape(i, Z, SH) * shape(i, Y, SH) * shape(i, X, NO);
    point_B.y() += B[g_z][g_y][g_x].y() * shape(i, Z, SH) * shape(i, Y, NO) * shape(i, X, SH);
    point_B.z() += B[g_z][g_y][g_x].z() * shape(i, Z, NO) * shape(i, Y, SH) * shape(i, X, SH);
  }}}
}


void Particles::push(const Vector3<PetscReal>& point_E, const Vector3<PetscReal>& point_B, Point& point) const {
  PetscReal alpha = 0.5 * dt * charge(point);
  PetscReal m = mass(point);

  Vector3<PetscReal>& r = point.r;
  Vector3<PetscReal>& p = point.p;

  const Vector3<PetscReal> w = p + point_E * alpha;

  PetscReal energy = sqrt(m * m + w.dot(w));

  const Vector3<PetscReal> h = point_B * alpha / energy;

  const Vector3<PetscReal> s = h * 2.0 / (1.0 + h.dot(h));

  p = point_E * alpha + w * (1.0 - h.dot(s)) + w.cross(s) + h * (s.dot(w));

  energy = sqrt(m * m + p.dot(p));

  r += p * dt / energy;

  if (geom_nx == 1) r.x() = 0.5;
  if (geom_ny == 1) r.y() = 0.5;
  if (geom_nz == 1) r.z() = 0.5;
}


PetscErrorCode Particles::communicate() {
  PetscFunctionBegin;
  constexpr PetscInt dim = 3;
  constexpr PetscInt neighbours_num = 27;

  std::vector<Point> outgoing[neighbours_num];
  std::vector<Point> incoming[neighbours_num];

  auto set_index = [&](const Vector3<PetscReal>& r, Vector3<PetscInt>& index, Axis axis) {
    index[axis] = (r[axis] < l_start[axis]) ? 0 : (r[axis] < l_end[axis]) ? 1 : 2;
  };
  PetscInt center_index = to_contiguous_index(1, 1, 1);

  auto end = points_.end();
  for (auto it = points_.begin(); it != end; ++it) {
    const Vector3<PetscReal>& r = it->r;
    Vector3<PetscInt> v_index;
    set_index(r, v_index, X);
    set_index(r, v_index, Y);
    set_index(r, v_index, Z);

    PetscInt index = to_contiguous_index(v_index[X], v_index[Y], v_index[Z]);
    if (index == center_index) continue;  // Particle didn't cross local boundaries

    outgoing[index].emplace_back(std::move(*it));
    std::swap(*it, *(end - 1));
    --it;
    --end;
  }
  points_.erase(end, points_.end());

  size_t o_num[neighbours_num];
  size_t i_num[neighbours_num];
  for (PetscInt i = 0; i < neighbours_num; ++i) {
    o_num[i] = outgoing[i].size();
    i_num[i] = 0;
  }

  MPI_Request reqs[2 * (neighbours_num - 1)];
  PetscInt req = 0;

  /// @note `PETSC_DEFAULT` is identical to `MPI_PROC_NULL`, so we can safely send/recv to/from neighbours.
  for (PetscInt s = 0; s < neighbours_num; ++s) {
    if (s == center_index) continue;
    PetscInt r = (neighbours_num - 1) - s;
    PetscCallMPI(MPI_Isend(&o_num[s], sizeof(size_t), MPI_BYTE, neighbours[s], MPI_TAG_NUMBERS, PETSC_COMM_WORLD, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(&i_num[r], sizeof(size_t), MPI_BYTE, neighbours[r], MPI_TAG_NUMBERS, PETSC_COMM_WORLD, &reqs[req++]));
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));
  assert(o_num[center_index] == 0);
  assert(i_num[center_index] == 0);

  req = 0;
  for (PetscInt s = 0; s < neighbours_num; ++s) {
    if (s == center_index) continue;
    PetscInt r = (neighbours_num - 1) - s;
    incoming[r].resize(i_num[r]);
    PetscCallMPI(MPI_Isend(outgoing[s].data(), o_num[s] * sizeof(Point), MPI_BYTE, neighbours[s], MPI_TAG_POINTS, PETSC_COMM_WORLD, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(incoming[r].data(), i_num[r] * sizeof(Point), MPI_BYTE, neighbours[r], MPI_TAG_POINTS, PETSC_COMM_WORLD, &reqs[req++]));
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));

  for (PetscInt i = 0; i < neighbours_num; ++i) {
    if (i == center_index) continue;
    points_.insert(points_.end(),
      std::make_move_iterator(incoming[i].begin()),
      std::make_move_iterator(incoming[i].end()));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}
