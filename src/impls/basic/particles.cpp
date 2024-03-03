#include "particles.h"

#include "src/impls/basic/simulation.h"

namespace basic {

Particles::Particles(const Simulation& simulation, const Particles_parameters& parameters)
  : simulation_(simulation), parameters_(parameters) {
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
  #pragma omp for schedule(monotonic: dynamic, OMP_CHUNK_SIZE)
  for (auto it = points_.begin(); it != points_.end(); ++it) {
    Vector3<PetscReal> r0 = it->r;
    Vector3<PetscReal> local_E = 0.0;
    Vector3<PetscReal> local_B = 0.0;

		interpolate(r0, local_E, local_B);
    push(*it, local_E, local_B);
  }
	PetscFunctionReturn(PETSC_SUCCESS);
}


enum Shift : PetscInt {
  NO = 0,  // shape[x - i]
  SH = 1   // shape[x - (i + 0.5)]
};

// `Vector3_dim` is used as a coordinate space dimensionality

#pragma omp declare simd linear(x, y, z: 1), notinbranch
constexpr PetscInt index(PetscInt x, PetscInt y, PetscInt z, PetscInt comp, PetscInt shift) {
  return (((z * shape_width + y) * shape_width + x) * Vector3_dim + comp) * 2 + shift;
}

void Particles::interpolate(const Vector3<PetscReal>& r0, Vector3<PetscReal>& local_E, Vector3<PetscReal>& local_B) const {
  thread_local static PetscReal shape[shape_width * shape_width * shape_width * Vector3_dim * 2];

  // Subtracting `shape_radius` to use indexing in range `[0, shape_width)`
  const PetscInt node_px = TO_STEP(r0.x(), dx) - shape_radius;
  const PetscInt node_py = TO_STEP(r0.y(), dy) - shape_radius;
  const PetscInt node_pz = TO_STEP(r0.z(), dz) - shape_radius;

  PetscInt node_gx, node_gy, node_gz;

  #pragma omp simd collapse(Vector3_dim)
	for(PetscInt z = 0; z < shape_width; ++z) {
  for(PetscInt y = 0; y < shape_width; ++y) {
	for(PetscInt x = 0; x < shape_width; ++x) {
      node_gx = node_px + x;
      node_gy = node_py + y;
      node_gz = node_pz + z;

      /// @todo check the size_s == 1
      shape[index(x, y, z, X, NO)] = shape_function(r0.x() / dx - node_gx);
      shape[index(x, y, z, Y, NO)] = shape_function(r0.y() / dy - node_gy);
      shape[index(x, y, z, Z, NO)] = shape_function(r0.z() / dz - node_gz);

      shape[index(x, y, z, X, SH)] = shape_function(r0.x() / dx - (node_gx + 0.5));
      shape[index(x, y, z, Y, SH)] = shape_function(r0.y() / dy - (node_gy + 0.5));
      shape[index(x, y, z, Z, SH)] = shape_function(r0.z() / dz - (node_gz + 0.5));
	}}}

  Vector3<PetscReal> ***E, ***B;
  PetscCallVoid(DMDAVecGetArrayRead(simulation_.da(), simulation_.E(), &E));
  PetscCallVoid(DMDAVecGetArrayRead(simulation_.da(), simulation_.B(), &B));

  // #pragma omp simd collapse(Vector3_dim)
	for(PetscInt z = 0; z < shape_width; ++z) {
  for(PetscInt y = 0; y < shape_width; ++y) {
	for(PetscInt x = 0; x < shape_width; ++x) {
		node_gx = node_px + x;
		node_gy = node_py + y;
		node_gz = node_pz + z;

		/// @todo convert it something smaller?
		local_E.x() += E[node_gz][node_gy][node_gx].x() * shape[index(x, y, z, Z, NO)] * shape[index(x, y, z, Y, NO)] * shape[index(x, y, z, X, SH)];
		local_E.y() += E[node_gz][node_gy][node_gx].y() * shape[index(x, y, z, Z, NO)] * shape[index(x, y, z, Y, SH)] * shape[index(x, y, z, X, NO)];
		local_E.z() += E[node_gz][node_gy][node_gx].z() * shape[index(x, y, z, Z, SH)] * shape[index(x, y, z, Y, NO)] * shape[index(x, y, z, X, NO)];

		local_B.x() += B[node_gz][node_gy][node_gx].x() * shape[index(x, y, z, Z, SH)] * shape[index(x, y, z, Y, SH)] * shape[index(x, y, z, X, NO)];
		local_B.y() += B[node_gz][node_gy][node_gx].y() * shape[index(x, y, z, Z, SH)] * shape[index(x, y, z, Y, NO)] * shape[index(x, y, z, X, SH)];
		local_B.z() += B[node_gz][node_gy][node_gx].z() * shape[index(x, y, z, Z, NO)] * shape[index(x, y, z, Y, SH)] * shape[index(x, y, z, X, SH)];
	}}}
}


void Particles::push(Point& point, const Vector3<PetscReal>& local_E, const Vector3<PetscReal>& local_B) const {
  PetscReal alpha = 0.5 * dt * charge(point);
  PetscReal m = mass(point);

  Vector3<PetscReal>& r = point.r;
  Vector3<PetscReal>& p = point.p;

  const Vector3<PetscReal> w = p + local_E * alpha;

  PetscReal energy = sqrt(m * m + w.dot(w));

  const Vector3<PetscReal> h = local_B * alpha / energy;

  const Vector3<PetscReal> s = h * 2.0 / (1.0 + h.dot(h));

  p = local_E * alpha + w * (1.0 - h.dot(s)) + w.cross(s) + h * (s.dot(w));

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
