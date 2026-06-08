#include "particles.h"
#include "src/algorithms/drift_kinetic_push.h"
#include "src/algorithms/implicit_drift_kinetic.h"
#include "src/impls/drift_kinetic/simulation.h"
#include "src/impls/eccapfim/particles.h" // for `cell_traversal`
#include "src/utils/geometries.h"
#include "src/utils/utils.h"

namespace drift_kinetic {

PointByField make_point_at_gc(
  const Point& point, const Vector3R& Bp, PetscReal mp)
{
  const Vector3R b = Bp.normalized();
  const PetscReal Bmag = Bp.length();
  const PetscReal p_par = point.p.dot(b);
  const PetscReal p_perp = point.p.transverse_to(Bp).length();
  const PetscReal mu = mp * p_perp * p_perp / (2.0 * Bmag);
  PointByField r(point.r, p_perp, p_par, mu);
  r.p = point.p;
  return r;
}

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : interfaces::Particles(simulation.world, parameters),
    dk_curr_storage(world.size.elements_product()),
    dk_prev_storage(world.size.elements_product()),
    simulation_(simulation)
{
  PetscMPIInt size;
  PetscCallAbort(PETSC_COMM_WORLD, MPI_Comm_size(PETSC_COMM_WORLD, &size));
  update_cells = (size == 1) //
    ? std::bind(std::mem_fn(&Particles::update_cells_seq), this)
    : std::bind(std::mem_fn(&Particles::update_cells_mpi), this);


  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da, &J));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da, &M));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da, &J_loc));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da, &M_loc));
}

PetscErrorCode Particles::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&J));
  PetscCall(VecDestroy(&M));
  PetscCall(VecDestroy(&J_loc));
  PetscCall(VecDestroy(&M_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::initialize_point_by_field(const Arr B_arr)
{
  PetscFunctionBeginUser;
  const PetscReal qm = parameters.q / parameters.m;
  const PetscReal mp = parameters.m;
  drift_kinetic::DriftKineticEsirkepov esirkepov(B_arr);

  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    auto& cell = storage[g];
    if (cell.empty())
      continue;

    auto& dk_cell = dk_curr_storage[g];
    dk_cell.clear();

    PetscInt i = 0;
    for (const auto& point : cell) {
      Vector3R B_p{};
      PetscCall(esirkepov.interpolate_B(B_p, point.r));
      if (coord_is_gc_)
        dk_cell.emplace_back(make_point_at_gc(point, B_p, mp));
      else
        dk_cell.emplace_back(point, B_p, mp, qm);
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal Particles::kinetic_energy_local() const
{
  PetscReal w = 0.0;
  const PetscReal mpw = parameters.n / static_cast<PetscReal>(parameters.Np);
  PetscCallAbort(PETSC_COMM_WORLD,
    DMGlobalToLocal(simulation_.da, simulation_.B, INSERT_VALUES, simulation_.B_loc));
  PetscCallAbort(PETSC_COMM_WORLD,
    DMDAVecGetArrayRead(simulation_.da, simulation_.B_loc, &simulation_.B_arr));

  drift_kinetic::DriftKineticEsirkepov esirkepov(simulation_.B_arr);
#pragma omp parallel for reduction(+ : w)
  for (auto&& cell : dk_curr_storage) {
    for (auto&& point : cell) {
      Vector3R B_p{};
      PetscCallAbort(PETSC_COMM_WORLD, esirkepov.interpolate_B(B_p, point.r));
      w += POW2(point.p_parallel) + 2.0 * point.mu_p * B_p.length();
    }
  }

  PetscCallAbort(PETSC_COMM_WORLD,
    DMDAVecRestoreArrayRead(simulation_.da, simulation_.B_loc, &simulation_.B_arr));
  return 0.5 * parameters.m * mpw * w;
}

PetscReal Particles::get_average_iteration_number() const
{
  return avgit;
}

PetscInt Particles::get_max_iteration_number() const
{
  return maxit;
}

PetscErrorCode Particles::form_iteration()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMDAVecGetArrayWrite(da, M_loc, &M_arr));

  avgit = 0.0;
  maxit = 0;

  PetscReal q = parameters.q;
  PetscReal m = parameters.m;

  const PetscReal inv_size = size > 0 ? 1.0 / static_cast<PetscReal>(size) : 0.0;

#pragma omp parallel for reduction(+ : avgit) reduction(max : maxit)
    for (PetscInt g = 0; g < (PetscInt)dk_curr_storage.size(); ++g) {
      const auto& prev_cell = dk_prev_storage[g];

      PetscInt i = 0;
      for (auto& curr : dk_curr_storage[g]) {
        auto prev(prev_cell[i]);

        DriftKineticPush push(q / m, m);
        drift_kinetic::DriftKineticEsirkepov util_local(E_arr, Bn_arr, B_arr, Bn1_arr, J_arr, M_arr);

        push.set_fields_callback(
          [&](const Vector3R& r0, const Vector3R& rn, Vector3R& E_p, Vector3R& B_p,
            Vector3R& gradB_p, Vector3R& rotB_p) { util_local.interpolate(E_p, B_p, gradB_p, rotB_p, rn, r0); });

        push.process(dt, curr, prev);

        avgit +=  (PetscReal)push.get_iteration_number() * inv_size;
        maxit = std::max(maxit, (PetscInt)push.get_iteration_number());

        const PetscReal a0 = qn_Np(curr);
        const PetscReal b0 = curr.mu_p * n_Np(curr);
        const Vector3R Vph = (curr.r - prev.r) / dt;
        curr.p = Vph;

        util_local.decomposition(curr.r, prev.r, Vph, a0, b0);

        ++i;
      }
    }

  PetscCall(DMDAVecRestoreArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, M_loc, &M_arr));
  PetscCall(DMLocalToGlobal(da, J_loc, ADD_VALUES, J));
  PetscCall(DMLocalToGlobal(da, M_loc, ADD_VALUES, M));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal Particles::n_Np(const PointByField& point) const
{
  Point dummy(point.r, Vector3R{});
  return interfaces::Particles::n_Np(dummy);
}

PetscReal Particles::qn_Np(const PointByField& point) const
{
  Point dummy(point.r, Vector3R{});
  return interfaces::Particles::qn_Np(dummy);
}

PetscErrorCode Particles::sync_dk_curr_storage()
{
  PetscFunctionBeginUser;
  const PetscReal qm = parameters.q / parameters.m;
  const PetscReal mp = parameters.m;

  PetscCall(DMGlobalToLocal(simulation_.da, simulation_.B, INSERT_VALUES, simulation_.B_loc));
  PetscCall(DMDAVecGetArrayRead(simulation_.da, simulation_.B_loc, &simulation_.B_arr));

  drift_kinetic::DriftKineticEsirkepov esirkepov(simulation_.B_arr);

  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    auto& cell = storage[g];
    if (cell.empty())
      continue;

    auto& dk_cell = dk_curr_storage[g];
    for (const auto& point : cell) {
      Vector3R B_p{};
      PetscCall(esirkepov.interpolate_B(B_p, point.r));
      if (coord_is_gc_)
        dk_cell.emplace_back(make_point_at_gc(point, B_p, mp));
      else
        dk_cell.emplace_back(point, B_p, mp, qm);
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(simulation_.da, simulation_.B_loc, &simulation_.B_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::prepare_storage()
{
  PetscFunctionBeginUser;
  size = 0;
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    if (auto& curr = dk_curr_storage[g]; !curr.empty()) {
      auto& prev = dk_prev_storage[g];
      prev = std::vector(curr.begin(), curr.end());
      size += (PetscInt)curr.size();
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::correct_coordinates(PointByField& point)
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

PetscErrorCode Particles::update_cells_seq()
{
  PetscFunctionBeginUser;
  PetscLogEventBegin(events[0], 0, 0, 0, 0);

  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    auto it = dk_curr_storage[g].begin();
    while (it != dk_curr_storage[g].end()) {
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
        dk_curr_storage[ng].emplace_back(std::move(*it));

      it = dk_curr_storage[g].erase(it);
    }
  }

  PetscLogEventEnd(events[0], 0, 0, 0, 0);

  PetscInt sum = 0;
  for (const auto& cell : dk_curr_storage)
    sum += cell.size();

  LOG("  Cells have been updated, total number of particles: {}", sum);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::update_cells_mpi()
{
  PetscFunctionBeginUser;
  constexpr PetscInt neighbors_num = POW3(3);
  constexpr PetscInt center_index = indexing::petsc_index(1, 1, 1, 0, 3, 3, 3, 1);

  auto get_index = [](const Vector3I& r, Axis axis, const World& world) {
    if (r[axis] < world.start[axis])
      return 0;
    if (r[axis] < world.end[axis])
      return 1;
    return 2;
  };

  auto get_neighbor = [](PetscInt i, const World& world) {
    return world.neighbors[i] < 0 ? MPI_PROC_NULL : world.neighbors[i];
  };

  std::vector<PointByField> outgoing[neighbors_num];
  std::vector<PointByField> incoming[neighbors_num];

  PetscLogEventBegin(events[0], 0, 0, 0, 0);

  LOG("  Starting MPI cells update for \"{}\"", parameters.sort_name);
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    Vector3I pg{
      world.start[X] + g % world.size[X],
      world.start[Y] + (g / world.size[X]) % world.size[Y],
      world.start[Z] + (g / world.size[X]) / world.size[Y],
    };

    auto it = dk_curr_storage[g].begin();
    while (it != dk_curr_storage[g].end()) {
      Vector3I ng{
        FLOOR_STEP(it->x(), dx),
        FLOOR_STEP(it->y(), dy),
        FLOOR_STEP(it->z(), dz),
      };

      if (pg[X] == ng[X] && pg[Y] == ng[Y] && pg[Z] == ng[Z]) {
        it = std::next(it);
        continue;
      }

      PetscInt i = indexing::petsc_index( //
        get_index(ng, X, world),          //
        get_index(ng, Y, world),          //
        get_index(ng, Z, world),          //
        0, 3, 3, 3, 1);

      if (i == center_index) {
        PetscInt j = world.s_g(   //
          ng[X] - world.start[X], //
          ng[Y] - world.start[Y], //
          ng[Z] - world.start[Z]);

        dk_curr_storage[j].emplace_back(std::move(*it));
        it = dk_curr_storage[g].erase(it);
        continue;
      }

      PetscCall(correct_coordinates(*it));

      outgoing[i].emplace_back(std::move(*it));
      it = dk_curr_storage[g].erase(it);
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
    PetscCallMPI(MPI_Isend(outgoing[s].data(), o_num[s] * sizeof(PointByField), MPI_BYTE,
      get_neighbor(s, world), MPI_TAG_POINTS, comm, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(incoming[r].data(), i_num[r] * sizeof(PointByField), MPI_BYTE,
      get_neighbor(r, world), MPI_TAG_POINTS, comm, &reqs[req++]));
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

      dk_curr_storage[g].emplace_back(std::move(point));
    }
  }

  PetscLogEventEnd(events[0], 0, 0, 0, 0);

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
  for (const auto& cell : dk_curr_storage)
    sum += cell.size();

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPIU_INT, MPI_SUM, comm));
  LOG("  Cells have been updated, total number of particles: {}", sum);
  PetscFunctionReturn(PETSC_SUCCESS);
}


}  // namespace drift_kinetic
