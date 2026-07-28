#include "particles.h"
#include "src/algorithms/drift_kinetic_push.h"
#include "src/algorithms/implicit_drift_kinetic.h"
#include "src/impls/drift_kinetic/simulation.h"
#include "src/impls/eccapfim/particles.h" // for `cell_traversal`
#include "src/utils/geometries.h"
#include "src/utils/utils.h"

namespace drift_kinetic {

namespace {

// Kahan compensated summation: `add()` folds the rounding error of each
// addition into `c` and subtracts it back on the next term, so the running
// sum stays accurate to ~eps regardless of how many terms are added (plain
// summation drifts to ~eps*N for N particles).
struct KahanSum {
  PetscReal sum = 0.0;
  PetscReal c = 0.0;

  void add(PetscReal value)
  {
    const PetscReal y = value - c;
    const PetscReal t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
};

#pragma omp declare reduction(kahan_add:KahanSum : omp_out.add(omp_in.sum)) \
  initializer(omp_priv = KahanSum{})

}  // namespace

PointByField make_point_at_gc(
  const Point& point, const Vector3R& Bp, PetscReal mp)
{
  const Vector3R b = Bp.normalized();
  const PetscReal Bmag = Bp.length();
  const PetscReal p_par = point.p.dot(b);
  const PetscReal p_perp = point.p.transverse_to(Bp).length();
  const PetscReal mu = mp * p_perp * p_perp / (2.0 * Bmag);
  PointByField r(point.r, p_par, mu);
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
  KahanSum w;
  const PetscReal mpw = parameters.n / static_cast<PetscReal>(parameters.Np);
  PetscCallAbort(PETSC_COMM_WORLD,
    DMGlobalToLocal(simulation_.da, simulation_.B, INSERT_VALUES, simulation_.B_loc));
  PetscCallAbort(PETSC_COMM_WORLD,
    DMDAVecGetArrayRead(simulation_.da, simulation_.B_loc, &simulation_.B_arr));

  drift_kinetic::DriftKineticEsirkepov esirkepov(simulation_.B_arr);
#pragma omp parallel for reduction(kahan_add : w)
  for (auto&& cell : dk_curr_storage) {
    for (auto&& point : cell) {
      Vector3R B_p{};
      PetscCallAbort(PETSC_COMM_WORLD, esirkepov.interpolate_B(B_p, point.r));
      w.add(POW2(point.p_parallel) + 2.0 * point.mu_p * B_p.length() / parameters.m);
    }
  }

  PetscCallAbort(PETSC_COMM_WORLD,
    DMDAVecRestoreArrayRead(simulation_.da, simulation_.B_loc, &simulation_.B_arr));
  return 0.5 * parameters.m * mpw * w.sum;
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

  constexpr PetscInt max_substep_depth = 4;

  KahanSum aud_push, aud_gradB, aud_total;
  PetscReal aud_max_push = 0.0;
  PetscReal aud_max_gradB = 0.0;

#pragma omp parallel for reduction(+ : avgit) reduction(max : maxit) \
  reduction(kahan_add : aud_push, aud_gradB, aud_total) \
  reduction(max : aud_max_push, aud_max_gradB)
    for (PetscInt g = 0; g < (PetscInt)dk_curr_storage.size(); ++g) {
      const auto& prev_cell = dk_prev_storage[g];

      PetscInt i = 0;
      for (auto& curr : dk_curr_storage[g]) {
        auto prev(prev_cell[i]);

        DriftKineticPush push(q / m, m);
        drift_kinetic::DriftKineticEsirkepov util_local(E_arr, Bn_arr, B_arr, Bn1_arr, J_arr, M_arr);

        push.set_fields_callback(
          [&](const Vector3R& r0, const Vector3R& rn, Vector3R& E_p, PetscReal& lenB_p, Vector3R& b_p,
            Vector3R& gradB_p, Vector3R& rotB_p) { util_local.interpolate(E_p, lenB_p, b_p, gradB_p, rotB_p, rn, r0); });

        PetscReal it_sum = 0.0;
        PetscInt it_max = 0;

        // Per-particle energy defects, accumulated over (sub)segments; see
        // `Particles::EnergyAudit` for the meaning of the three levels.
        PetscReal D_push_p = 0.0;
        PetscReal D_gradB_p = 0.0;
        PetscReal D_total_p = 0.0;

        std::function<void(PetscReal, PointByField&, const PointByField&, PetscInt)> solve_segment =
          [&](PetscReal dt_sub, PointByField& pn, const PointByField& p0, PetscInt depth) {
            push.process(dt_sub, pn, p0);

            it_sum += (PetscReal)push.get_iteration_number();
            it_max = std::max(it_max, (PetscInt)push.get_iteration_number());

            if (!push.has_converged() && depth < max_substep_depth) {
              PointByField mid(p0);
              mid.r = 0.5 * (p0.r + pn.r);
              mid.p_parallel = 0.5 * (p0.p_parallel + pn.p_parallel);

              const PetscReal dt_half = 0.5 * dt_sub;
              solve_segment(dt_half, mid, p0, depth + 1);
              solve_segment(dt_half, pn, mid, depth + 1);
              return;
            }

            const PetscReal a0 = qn_Np(pn);
            const PetscReal b0 = pn.mu_p * n_Np(pn);
            const Vector3R Vp_sub = (pn.r - p0.r) / dt_sub;

            util_local.decomposition(pn.r, p0.r, Vp_sub, a0 * (dt_sub / dt), b0 * (dt_sub / dt));

            if (energy_audit_enabled_) {
              // Per-(sub)segment defects of the discrete energy identities of
              // the scheme; see `Particles::EnergyAudit` for the three levels.
              drift_kinetic::DriftKineticEsirkepov::EndpointB f;
              PetscCallAbort(PETSC_COMM_WORLD,
                util_local.interpolate_B_endpoints(f, pn.r, p0.r));

              // hatb = b/|b|^2 with b = (b^n + b^{n+1})/2 — the same vector
              // the gradB rule and the magnetization deposit use.
              const Vector3R b_mid =
                0.5 * (f.Bn_0.normalized() + f.Bn1_n.normalized());
              const Vector3R hatb = b_mid / b_mid.squared();

              const PetscReal dK_par =
                0.5 * m * (POW2(pn.p_parallel) - POW2(p0.p_parallel));
              const PetscReal dK_mu =
                pn.mu_p * (f.Bn1_n.length() - f.Bn_0.length());
              const PetscReal W_E = q * dt_sub * push.get_Eh().dot(Vp_sub);
              const PetscReal W_gradB =
                pn.mu_p * dt_sub * Vp_sub.dot(push.get_gradBh());
              // Right-hand side of the gradB interpolation rule:
              // hatb . (B^{n+1/2}(R1) - B^{n+1/2}(R0)).
              const PetscReal W_hB = pn.mu_p * hatb.dot(f.Bnh_n - f.Bnh_0);
              // Particle share of M^{n+1/2} . (B^{n+1} - B^n) as deposited:
              // half of mu*hatb at each endpoint, weighted like the deposit.
              const PetscReal W_M = 0.5 * pn.mu_p * (dt_sub / dt) *
                hatb.dot((f.Bn1_0 - f.Bn_0) + (f.Bn1_n - f.Bn_n));

              const PetscReal wt = n_Np(pn);
              D_push_p += wt * (dK_par + W_gradB - W_E);
              D_gradB_p += wt * (W_gradB - W_hB);
              D_total_p += wt * (dK_par + dK_mu - W_E - W_M);
            }
          };

        solve_segment(dt, curr, prev, 0);

        avgit += it_sum * inv_size;
        maxit = std::max(maxit, it_max);

        if (energy_audit_enabled_) {
          aud_push.add(D_push_p);
          aud_gradB.add(D_gradB_p);
          aud_total.add(D_total_p);
          aud_max_push = std::max(aud_max_push, std::abs(D_push_p));
          aud_max_gradB = std::max(aud_max_gradB, std::abs(D_gradB_p));
        }

        curr.p = (curr.r - prev.r) / dt;

        ++i;
      }
    }

  if (energy_audit_enabled_) {
    audit_.D_push = aud_push.sum;
    audit_.D_gradB = aud_gradB.sum;
    audit_.D_total = aud_total.sum;
    audit_.max_D_push = aud_max_push;
    audit_.max_D_gradB = aud_max_gradB;
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

/*
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
*/

PetscErrorCode Particles::prepare_storage()
{
  PetscFunctionBeginUser;

  size = 0;

  PetscCheck(
    dk_curr_storage.size() == dk_prev_storage.size(),
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_SIZ,
    "DK current and previous storages have different sizes: %zu != %zu",
    dk_curr_storage.size(),
    dk_prev_storage.size());

  for (PetscInt g = 0;
       g < static_cast<PetscInt>(dk_curr_storage.size());
       ++g) {
    const auto& curr = dk_curr_storage[g];
    auto& prev = dk_prev_storage[g];

    prev.assign(curr.begin(), curr.end());

    size += static_cast<PetscInt>(curr.size());
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::restore_from_prev_storage()
{
  PetscFunctionBeginUser;

  PetscCheck(
    dk_curr_storage.size() == dk_prev_storage.size(),
    PETSC_COMM_WORLD,
    PETSC_ERR_ARG_SIZ,
    "DK current and previous storages have different sizes: %zu != %zu",
    dk_curr_storage.size(),
    dk_prev_storage.size());

  PetscInt restored_size = 0;

  for (PetscInt g = 0;
       g < static_cast<PetscInt>(dk_prev_storage.size());
       ++g) {
    auto& curr = dk_curr_storage[g];
    const auto& prev = dk_prev_storage[g];

    curr.assign(prev.begin(), prev.end());

    restored_size += static_cast<PetscInt>(prev.size());
  }

  PetscCheck(
    restored_size == size,
    PETSC_COMM_WORLD,
    PETSC_ERR_PLIB,
    "Restored DK particle number differs from saved number: "
    "restored=%" PetscInt_FMT ", saved=%" PetscInt_FMT,
    restored_size,
    size);

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
