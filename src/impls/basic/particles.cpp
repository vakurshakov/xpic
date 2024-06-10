#include "particles.h"

#include "src/impls/basic/simulation.h"
#include "src/impls/simple_interpolation.h"

namespace basic {

Particles::Particles(Simulation& simulation, const Particles_parameters& parameters)
  : simulation_(simulation) {
  PetscFunctionBeginUser;
  parameters_ = parameters;

  DM& da = simulation_.da_;
  PetscCallVoid(DMDAGetNeighbors(da, &neighbours));

  Vector3I start, size;
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

  /// @note This local current is local to each particle! It's can be useful for diagnosing it.
  PetscCallVoid(DMCreateLocalVector(da, &local_J));
  PetscFunctionReturnVoid();
}

Particles::~Particles() {
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&local_J));
  PetscFunctionReturnVoid();
}


PetscErrorCode Particles::add_particle(const Point& point) {
  PetscFunctionBeginUser;
  const Vector3R& r = point.r;
  /// @todo This check should be moved, probably, into particle initializer
  if (l_start.x() <= r.x() && r.x() < l_end.x() &&
      l_start.y() <= r.y() && r.y() < l_end.y() &&
      l_start.z() <= r.z() && r.z() < l_end.z()) {
    #pragma omp critical
    points_.emplace_back(point);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::push() {
  PetscFunctionBeginUser;
  const DM& da = simulation_.da_;
  PetscCall(DMGetLocalVector(da, &local_E));
  PetscCall(DMGetLocalVector(da, &local_B));

  PetscCall(DMGlobalToLocal(da, simulation_.E_, INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, simulation_.B_, INSERT_VALUES, local_B));
  PetscCall(VecSet(local_J, 0.0));

  PetscCall(DMDAVecGetArrayRead(da, local_E, &E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &B));
  PetscCall(DMDAVecGetArrayWrite(da, local_J, &J));

  #pragma omp for schedule(monotonic: dynamic, OMP_CHUNK_SIZE)
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    Vector3R point_E = 0.0;
    Vector3R point_B = 0.0;

    const Node node(it->r);

    static Shape shape[2];
    #pragma omp threadprivate(shape)

    fill_shape(node.g, node.r, l_width, false, shape[0]);
    fill_shape(node.g, node.r, l_width, true, shape[1]);
    interpolate(node.g, shape[0], shape[1], point_E, point_B);

    push(point_E, point_B, *it);

    const Node new_node(it->r);

    fill_shape(new_node.g, node.r, l_width, false, shape[0]);
    fill_shape(new_node.g, new_node.r, l_width, false, shape[1]);
    decompose(new_node.g, shape[0], shape[1], *it);
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &B));
  PetscCall(DMDAVecRestoreArrayWrite(da, local_J, &J));

  PetscCall(DMLocalToGlobal(da, local_J, ADD_VALUES, simulation_.J_));

  PetscCall(DMRestoreLocalVector(da, &local_E));
  PetscCall(DMRestoreLocalVector(da, &local_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}


void Particles::interpolate(const Vector3I& p_g, Shape& no, Shape& sh, Vector3R& point_E, Vector3R& point_B) const {
  Simple_interpolation interpolation(l_width, no, sh);
  interpolation.process(p_g, {{point_E, E}}, {{point_B, B}});
}


void Particles::push(const Vector3R& point_E, const Vector3R& point_B, Point& point) const {
  PetscReal alpha = 0.5 * dt * charge(point);
  PetscReal m = mass(point);

  Vector3R& r = point.r;
  Vector3R& p = point.p;

  const Vector3R w = p + point_E * alpha;
  PetscReal energy = sqrt(m * m + w.dot(w));

  const Vector3R h = point_B * alpha / energy;
  const Vector3R s = h * 2.0 / (1.0 + h.dot(h));
  p = point_E * alpha + w * (1.0 - h.dot(s)) + w.cross(s) + h * (s.dot(w));

  energy = sqrt(m * m + p.dot(p));
  r += p * dt / energy;

  if (geom_nx == 1) r.x() = 0.5;
  if (geom_ny == 1) r.y() = 0.5;
  if (geom_nz == 1) r.z() = 0.5;
}


void Particles::decompose(const Vector3I& p_g, Shape& old_shape, Shape& new_shape, const Point& point) {
  const PetscReal alpha = charge(point) * density(point) / particles_number(point) / (6.0 * dt);
  const PetscReal qx = alpha * dx;
  const PetscReal qy = alpha * dy;
  const PetscReal qz = alpha * dz;

  auto compute_Jx = [&](PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jx) {
    PetscInt i = ((z * shape_width + y) * shape_width + x);
    PetscInt j = (z * shape_width + y);

    PetscReal p_wx = - qx * (new_shape(i, X) - old_shape(i, X)) * (
      new_shape(i, Y) * (2.0 * new_shape(i, Z) + old_shape(i, Z)) +
      old_shape(i, Y) * (2.0 * old_shape(i, Z) + new_shape(i, Z)));

    temp_jx[j] = ((x > 0) * temp_jx[j]) + p_wx;
    return temp_jx[j];
  };

  auto compute_Jy = [&](PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jy) {
    PetscInt i = ((z * shape_width + y) * shape_width + x);
    PetscInt j = (z * shape_width + x);

    PetscReal p_wy = - qy * (new_shape(i, Y) - old_shape(i, Y)) * (
      new_shape(i, X) * (2.0 * new_shape(i, Z) + old_shape(i, Z)) +
      old_shape(i, X) * (2.0 * old_shape(i, Z) + new_shape(i, Z)));

    temp_jy[j] = ((y > 0) * temp_jy[j]) + p_wy;
    return temp_jy[j];
  };

  auto compute_Jz = [&](PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jz) {
    PetscInt i = ((z * shape_width + y) * shape_width + x);
    PetscInt j = (y * shape_width + x);

    PetscReal p_wz = - qz * (new_shape(i, Z) - old_shape(i, Z)) * (
      new_shape(i, Y) * (2.0 * new_shape(i, X) + old_shape(i, X)) +
      old_shape(i, Y) * (2.0 * old_shape(i, X) + new_shape(i, X)));

    temp_jz[j] = ((z > 0) * temp_jz[j]) + p_wz;
    return temp_jz[j];
  };

  decompose_dir(p_g, compute_Jx, X);
  decompose_dir(p_g, compute_Jy, Y);
  decompose_dir(p_g, compute_Jz, Z);
}

void Particles::decompose_dir(const Vector3I& p_g, const Compute_j& compute_j, Axis dir) {
  static PetscReal temp_j[shape_width * shape_width];
  #pragma omp threadprivate(temp_j)

  PetscInt g_x, g_y, g_z;

  for (PetscInt z = 0; z < l_width[Z]; ++z) {
  for (PetscInt y = 0; y < l_width[Y]; ++y) {
  for (PetscInt x = 0; x < l_width[X]; ++x) {
    g_x = p_g[X] + x;
    g_y = p_g[Y] + y;
    g_z = p_g[Z] + z;

    PetscReal p_j = compute_j(x, y, z, temp_j);

    #pragma omp atomic update
    J[g_z][g_y][g_x][dir] += p_j;
  }}}
}


PetscErrorCode Particles::communicate() {
  PetscFunctionBeginUser;
  constexpr PetscInt dim = 3;
  constexpr PetscInt neighbours_num = 27;

  std::vector<Point> outgoing[neighbours_num];
  std::vector<Point> incoming[neighbours_num];

  auto set_index = [&](const Vector3R& r, Vector3I& index, Axis axis) {
    index[axis] = (r[axis] < l_start[axis]) ? 0 : (r[axis] < l_end[axis]) ? 1 : 2;
  };

  auto correct_coordinates = [&](Point& point) {
    if (simulation_.bounds_[X] == DM_BOUNDARY_PERIODIC) g_bound_periodic(point, X);
    if (simulation_.bounds_[Y] == DM_BOUNDARY_PERIODIC) g_bound_periodic(point, Y);
    if (simulation_.bounds_[Z] == DM_BOUNDARY_PERIODIC) g_bound_periodic(point, Z);
  };

  PetscInt center_index = to_contiguous_index(1, 1, 1);

  auto end = points_.end();
  for (auto it = points_.begin(); it != end; ++it) {
    const Vector3R& r = it->r;
    Vector3I v_index;
    set_index(r, v_index, X);
    set_index(r, v_index, Y);
    set_index(r, v_index, Z);

    PetscInt index = to_contiguous_index(v_index[X], v_index[Y], v_index[Z]);
    if (index == center_index) continue;  // Particle didn't cross local boundaries

    correct_coordinates(*it);
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
