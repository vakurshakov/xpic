#include "particles.h"
#include "src/algorithms/drift_kinetic_push.h"
#include "src/algorithms/drift_kinetic_implicit.h"
#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/geometries.h"
#include "src/utils/utils.h"
#include "src/impls/eccapfim/cell_traversal.h"

namespace drift_kinetic {

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
  DriftKineticEsirkepov esirkepov(nullptr, B_arr, nullptr, nullptr);

  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    auto& cell = storage[g];
    if (cell.empty())
      continue;

    auto& dk_cell = dk_curr_storage[g];
    dk_cell.clear();

    for (const auto& point : cell) {
      Vector3R B_p{};
      PetscCall(esirkepov.interpolate_B(B_p, point.r));

      dk_cell.emplace_back(point, B_p, mp, qm);
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::form_iteration()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMDAVecGetArrayWrite(da, M_loc, &M_arr));
  PetscCall(DMDAVecGetArrayWrite(da, simulation_.dBdx_loc, &simulation_.dBdx_arr));
  PetscCall(DMDAVecGetArrayWrite(da, simulation_.dBdy_loc, &simulation_.dBdy_arr));
  PetscCall(DMDAVecGetArrayWrite(da, simulation_.dBdz_loc, &simulation_.dBdz_arr));

  PetscReal q = parameters.q;
  PetscReal m = parameters.m;

  PetscReal xb = (world.start[X] - 0.5) * dx;
  PetscReal yb = (world.start[Y] - 0.5) * dy;
  PetscReal zb = (world.start[Z] - 0.5) * dz;

  PetscReal xe = (world.end[X] + 0.5) * dx;
  PetscReal ye = (world.end[Y] + 0.5) * dy;
  PetscReal ze = (world.end[Z] + 0.5) * dz;

  static const PetscReal max = std::numeric_limits<double>::max();

  auto process_bound = [&](PetscReal vh, PetscReal x, PetscReal xb, PetscReal xe) {
    if (vh > 0)
      return (xe - x) / vh;
    else if (vh < 0)
      return (xb - x) / vh;
    else
      return max;
  };

  DriftKineticEsirkepov util(E_arr, B_arr, J_arr, M_arr);
  util.set_dBidrj_precomputed(simulation_.dBdx_arr, simulation_.dBdy_arr, simulation_.dBdz_arr);

  for (PetscInt g = 0; g < (PetscInt)dk_curr_storage.size(); ++g) {
    for (PetscInt i = 0; auto& curr : dk_curr_storage[g]) {
      const auto& prev(dk_prev_storage[g][i]);

      /// @todo this part should reuse the logic from:
      /// tests/drift_kinetic_push/drift_kinetic_push.h:620 implicit_test_utils::interpolation_test()
#if 1
      DriftKineticPush push(q / m, m);
      push.set_fields_callback(  //
        [&](const Vector3R& rn, const Vector3R& r0, Vector3R& E_p, Vector3R& B_p,
          Vector3R& gradB_p) {
          E_p = {};
          B_p = {};
          gradB_p = {};
          Vector3R Es_p, Bs_p, gradBs_p;
          Vector3R E_dummy, B_dummy, gradB_dummy;

          Vector3R pos = (rn - r0);
          auto coords = cell_traversal(rn, r0);
          PetscInt segments = (PetscInt)coords.size() - 1;

          if (segments <= 0)
            segments = 1;

          pos[X] = pos[X] != 0 ? pos[X] / dx : (PetscReal)segments;
          pos[Y] = pos[Y] != 0 ? pos[Y] / dy : (PetscReal)segments;
          pos[Z] = pos[Z] != 0 ? pos[Z] / dz : (PetscReal)segments;

          util.interpolate(E_dummy, B_p, gradB_dummy, rn, r0);

          for (PetscInt s = 1; s < (PetscInt)coords.size(); ++s) {
            auto&& rs0 = coords[s - 1];
            auto&& rsn = coords[s - 0];
            util.interpolate(Es_p, B_dummy, gradBs_p, rsn, rs0);

            E_p += Es_p;
            gradB_p += gradBs_p;
          }

          E_p = E_p.elementwise_division(pos);
          gradB_p = gradB_p.elementwise_division(pos);
        });

      for (PetscReal dtau = 0.0, tau = 0.0; tau < dt; tau += dtau) {
        PetscReal dtx = process_bound((curr.x()-prev.x())/(dt - tau), curr.x(), xb, xe);
        PetscReal dty = process_bound((curr.y()-prev.y())/(dt - tau), curr.y(), yb, ye);
        PetscReal dtz = process_bound((curr.z()-prev.z())/(dt - tau), curr.z(), zb, ze);

        dtau = std::min({dt - tau, dtx, dty, dtz});

        push.process(dtau, curr, prev);

        auto coords = cell_traversal(curr.r, prev.r);
        PetscInt segments = (PetscInt)coords.size() - 1;
        if (segments <= 0)
          segments = 1;

        Vector3R Vp = dtau > 0 ? (curr.r - prev.r) / dtau : Vector3R{};
        PetscReal q_p = q*m/parameters.Np;

        Vector3R pos = (curr.r - prev.r);
        pos[X] = pos[X] != 0 ? pos[X] / dx : (PetscReal)segments;
        pos[Y] = pos[Y] != 0 ? pos[Y] / dy : (PetscReal)segments;
        pos[Z] = pos[Z] != 0 ? pos[Z] / dz : (PetscReal)segments;

        for (PetscInt s = 1; s < (PetscInt)coords.size(); ++s) {
          auto&& rs0 = coords[s - 1];
          auto&& rsn = coords[s - 0];
          util.decomposition_J(rsn, rs0, Vp.elementwise_division(pos), q_p);
        }

        util.decomposition_M(curr.r, curr.mu_p);
  #endif
        i++;
      }
  }
}

  PetscCall(DMDAVecRestoreArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, M_loc, &M_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, simulation_.dBdx_loc, &simulation_.dBdx_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, simulation_.dBdy_loc, &simulation_.dBdy_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, simulation_.dBdz_loc, &simulation_.dBdz_arr));
  PetscCall(DMLocalToGlobal(da, J_loc, ADD_VALUES, J));
  PetscCall(DMLocalToGlobal(da, M_loc, ADD_VALUES, M));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::prepare_storage()
{
  PetscFunctionBeginUser;
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    if (auto& curr = dk_curr_storage[g]; !curr.empty()) {
      auto& prev = dk_prev_storage[g];
      prev = std::vector(curr.begin(), curr.end());
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
        get_index(ng, X, world),           //
        get_index(ng, Y, world),           //
        get_index(ng, Z, world),           //
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
