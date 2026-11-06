#include "simulation.h"

#include "src/impls/hybrid_kinetic/diagnostic.h"
#include "src/utils/configuration.h"
#include "src/utils/utils.h"

namespace hybrid_kinetic {

namespace {

// Block of the mass matrix collected from the particles of one cell `g`. The
// component c of E is interpolated with S1 along c (half-integer nodes
// g-1..g+1) and with S2 across c (integer nodes g-1..g+2), so the block has
// 3 * 4 * 4 slots per component.
constexpr PetscInt mm_slots = 3 * 4 * 4;
constexpr PetscInt mm_rows = 3 * mm_slots;
constexpr PetscInt mm_block = mm_rows * mm_rows;

constexpr PetscInt mm_width(PetscInt c, PetscInt a)
{
  return c == a ? 3 : 4;
}

constexpr PetscInt mm_slot(PetscInt c, PetscInt i, PetscInt j, PetscInt k)
{
  return c * mm_slots + (k * mm_width(c, Y) + j) * mm_width(c, X) + i;
}

// Adds a particle at `r` of the cell `g` with the tensor `T` to `block`.
void add_to_mass_block(PetscReal* block, const Vector3I& g, const Vector3R& r,
  const PetscReal T[3][3])
{
  PetscReal along[3][3], across[3][4];
  for (Axis a : {X, Y, Z}) {
    // Position from the node g - 1, kept in the cell (the particle may leave
    // it during the iterations or be wrapped by the periodic boundary).
    const PetscReal s = std::clamp(r[a] / Dx[a] - g[a], 0.0, 1.0 - 1e-12) + 1;
    for (PetscInt i = 0; i < 3; i++)
      along[a][i] = spline_of_1st_order(s - (i + 0.5));
    for (PetscInt i = 0; i < 4; i++)
      across[a][i] = spline_of_2nd_order(s - i);
  }

  PetscInt n[3] = {0, 0, 0};
  PetscInt slot[3][mm_slots];
  PetscReal w[3][mm_slots];

  for (PetscInt c = 0; c < 3; c++) {
    auto weight = [&](PetscInt a, PetscInt i)
    {
      return a == c ? along[a][i] : across[a][i];
    };

    for (PetscInt k = 0; k < mm_width(c, Z); k++)
      for (PetscInt j = 0; j < mm_width(c, Y); j++)
        for (PetscInt i = 0; i < mm_width(c, X); i++) {
          const PetscReal wc = weight(X, i) * weight(Y, j) * weight(Z, k);
          if (wc == 0)
            continue;
          slot[c][n[c]] = mm_slot(c, i, j, k);
          w[c][n[c]++] = wc;
        }
  }

  for (PetscInt c1 = 0; c1 < 3; c1++)
    for (PetscInt c2 = 0; c2 < 3; c2++)
      for (PetscInt a = 0; a < n[c1]; a++) {
        PetscReal* row = block + slot[c1][a] * mm_rows;
        const PetscReal wa = T[c1][c2] * w[c1][a];
        for (PetscInt b = 0; b < n[c2]; b++)
          row[slot[c2][b]] += wa * w[c2][b];
      }
}

}  // namespace

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  PetscCall(drift_kinetic::Simulation::initialize_implementation());
  PetscCall(SNESSetFunction(snes, NULL, Simulation::form_iteration, this));
  PetscCall(init_kinetic_particles());
  PetscCall(init_preconditioner());

  // The base adds the drift-kinetic `EnergyConservation` last,
  // it misses the kinetic sorts.
  PetscCheck(dynamic_cast<drift_kinetic::EnergyConservation*>(diagnostics_.back().get()), PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE,
    "The last diagnostic of drift_kinetic::Simulation is expected to be EnergyConservation");

  diagnostics_.back() = std::make_unique<EnergyConservation>(*this);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::finalize()
{
  PetscFunctionBeginUser;
  if (use_mass_matrix) {
    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&matL));
    PetscCall(MatDestroy(&matA));
  }
  PetscCall(drift_kinetic::Simulation::finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::timestep_implementation(PetscInt t)
{
  PetscFunctionBeginUser;
  for (auto& sort : particles_)
    PetscCall(sort->prepare_storage());
  for (auto& sort : kinetic_particles_)
    PetscCall(sort->prepare_storage());

  mass_matrix_is_current = false;

  if (!evolve_fields) {
    PetscCall(VecCopy(E, E_hk));
    PetscCall(VecCopy(B, B_hk));
    PetscCall(form_current());
    PetscCall(accept_push(t));
  }
  else {
    PetscCall(to_snes(E, B, sol));

    PetscCall(SNESSolve(snes, NULL, sol));
    PetscCall(SNESGetIterationNumber(snes, &snes_iteration_number));

    LOG("SNESSolve() has finished, SNESConvergedReasonView():");
    PetscCall(SNESConvergedReasonView(snes, PETSC_VIEWER_STDOUT_WORLD));

    SNESConvergedReason reason;
    PetscCall(SNESGetConvergedReason(snes, &reason));

    PetscCheck(reason >= 0, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED,
      "SNESSolve has not converged, reason: %s", SNESConvergedReasons[reason]);

    PetscCall(SNESGetSolution(snes, &sol));
    PetscCall(from_snes(sol, E_hk, B_hk));

    // `SNESNGMRES` may return a point other than the last pushed one,
    // then the particles and sources are made consistent with it.
    PetscBool is_last;
    PetscCall(VecEqual(sol, last_push_fields, &is_last));

    if (!is_last) {
      LOG("The returned solution is not the last pushed point, pushing again");
      PetscCall(form_current());
    }

    PetscCall(accept_push(t));
    PetscCall(VecAXPBY(E, 2, -1, E_hk));
    PetscCall(VecAXPBY(B, 2, -1, B_hk));
  }

  for (auto& sort : particles_)
    PetscCall(sort->update_cells());
  for (auto& sort : kinetic_particles_)
    PetscCall(sort->update_cells());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_iteration(SNES, Vec vx, Vec vf, void* ctx)
{
  PetscFunctionBeginUser;
  auto* simulation = (Simulation*)ctx;
  PetscCall(simulation->from_snes(vx, simulation->E_hk, simulation->B_hk));
  PetscCall(simulation->form_current());
  PetscCall(simulation->form_function(vf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// All sorts are created by the base as drift-kinetic ones,
// the "kinetic" ones are rebuilt here with the same parameters.
PetscErrorCode Simulation::init_kinetic_particles()
{
  PetscFunctionBeginUser;
  const Configuration::json_t& json = CONFIG().json;
  auto it = json.find("Particles");

  if (it == json.end())
    PetscFunctionReturn(PETSC_SUCCESS);

  auto& all_particles = interfaces::Simulation::particles_;

  for (auto&& info : *it) {
    const std::string model = info.value("model", "drift_kinetic");

    PetscCheck(model == "kinetic" || model == "drift_kinetic", PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "Unknown particles model \"%s\", expected \"kinetic\" or \"drift_kinetic\"", model.c_str());

    if (model != "kinetic")
      continue;

    const std::string name = info.at("sort_name");
    auto is_named = [&](const auto& sort)
    {
      return sort->parameters.sort_name == name;
    };

    auto dk = std::find_if(particles_.begin(), particles_.end(), is_named);
    auto base =
      std::find_if(all_particles.begin(), all_particles.end(), is_named);

    auto sort = std::make_shared<KineticParticles>(world, (*dk)->parameters);
    PetscCall((*dk)->finalize());
    particles_.erase(dk);
    *base = sort;
    kinetic_particles_.emplace_back(sort);

    LOG("  {} are described by the kinetic model", name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_current()
{
  PetscFunctionBeginUser;
  // Guiding centers: zeroes `J`, `M` and moves `E_hk`, `B_hk` to locals.
  PetscCall(drift_kinetic::Simulation::form_current());

  PetscCall(DMDAVecGetArrayRead(da, E_loc, &E_arr));
  PetscCall(DMDAVecGetArrayRead(da, B_loc, &B_arr));

  for (auto& sort : kinetic_particles_) {
    PetscCall(VecSet(sort->J, 0));
    PetscCall(VecSet(sort->J_loc, 0));

    sort->E_arr = E_arr;
    sort->B_arr = B_arr;
    PetscCall(sort->form_iteration());
    PetscCall(VecAXPY(J, 1, sort->J));

    LOG("Push \"{}\": avg it={}", sort->parameters.sort_name, sort->get_average_iteration_number());
  }

  PetscCall(DMDAVecRestoreArrayRead(da, E_loc, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, B_loc, &B_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The field iteration is preconditioned by the linearized Maxwell equations
// with the mass matrix of the particles, see
// tests/drift_kinetic/field_preconditioner.md for the derivation.
PetscErrorCode Simulation::init_preconditioner()
{
  PetscFunctionBeginUser;
  const std::string name = CONFIG().json.value("FieldPreconditioner", "none");

  PetscCheck(name == "none" || name == "mass_matrix", PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
    "Unknown \"FieldPreconditioner\": \"%s\", expected \"none\" or \"mass_matrix\"", name.c_str());

  if (name == "none")
    PetscFunctionReturn(PETSC_SUCCESS);

  use_mass_matrix = true;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-npc_mass_matrix_every_iteration", &mass_matrix_every_iteration, nullptr));
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-npc_mass_matrix_check", &mass_matrix_check, nullptr));

  LOG("  Field preconditioner (FieldPreconditioner): {}, every iteration: {}", name, mass_matrix_every_iteration);

  PetscCall(SNESGetNPC(snes, &npc));
  PetscCall(SNESSetType(npc, SNESSHELL));
  PetscCall(SNESShellSetContext(npc, this));
  PetscCall(SNESShellSetSolve(npc, Simulation::form_preconditioned_step));
  PetscCall(SNESSetNPCSide(snes, PC_RIGHT));

  // COO indices of the cell blocks in the ghosted local numbering of `da`.
  const PetscInt cells = world.size.elements_product();
  std::vector<PetscInt> coo_i(cells * mm_block), coo_j(cells * mm_block);

#pragma omp parallel for
  for (PetscInt g = 0; g < cells; g++) {
    const Vector3I vg{
      world.start[X] + g % world.size[X],
      world.start[Y] + (g / world.size[X]) % world.size[Y],
      world.start[Z] + (g / world.size[X]) / world.size[Y],
    };

    PetscInt index[mm_rows];
    for (PetscInt c = 0; c < 3; c++)
      for (PetscInt k = 0; k < mm_width(c, Z); k++)
        for (PetscInt j = 0; j < mm_width(c, Y); j++)
          for (PetscInt i = 0; i < mm_width(c, X); i++) {
            const Vector3I node{vg[X] - 1 + i, vg[Y] - 1 + j, vg[Z] - 1 + k};
            index[mm_slot(c, i, j, k)] =
              (((node[Z] - world.gstart[Z]) * world.gsize[Y] +
                 (node[Y] - world.gstart[Y])) *
                  world.gsize[X] +
                (node[X] - world.gstart[X])) *
                3 +
              c;
          }

    for (PetscInt a = 0; a < mm_rows; a++)
      for (PetscInt b = 0; b < mm_rows; b++) {
        coo_i[g * mm_block + a * mm_rows + b] = index[a];
        coo_j[g * mm_block + a * mm_rows + b] = index[b];
      }
  }

  coo_v.resize(cells * mm_block);
  PetscCall(DMCreateMatrix(da, &matL));
  PetscCall(MatSetPreallocationCOOLocal(matL, (PetscCount)coo_v.size(), coo_i.data(), coo_j.data()));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "npc_ksp_"));
  PetscCall(KSPSetType(ksp, KSPGMRES));
  PetscCall(KSPSetTolerances(ksp, 1e-6, PETSC_CURRENT, PETSC_CURRENT, 1000));
  PetscCall(KSPSetFromOptions(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Shell solve of `npc`: one preconditioned step x <- x + dx, where dx solves
// the linearized equations F(x) + F'(x) dx = 0 with F' replaced by
// P = (matM + matL in E, rot in B). On entry the function of `npc` holds F(x),
// set by `SNESNGMRES`; on exit it holds the exact F of the new x, which
// `SNESNGMRES` takes as F(x^M).
PetscErrorCode Simulation::form_preconditioned_step(SNES npc, Vec vx)
{
  PetscFunctionBeginUser;
  Simulation* simulation;
  PetscCall(SNESShellGetContext(npc, &simulation));
  Simulation& s = *simulation;

  Vec vf, F_E, F_B, rhs, dE;
  PetscCall(SNESGetFunction(npc, &vf, nullptr, nullptr));
  PetscCall(DMGetGlobalVector(s.da, &F_E));
  PetscCall(DMGetGlobalVector(s.da, &F_B));
  PetscCall(DMGetGlobalVector(s.da, &rhs));
  PetscCall(DMGetGlobalVector(s.da, &dE));

  PetscCall(s.from_snes(vx, s.E_hk, s.B_hk));
  PetscCall(s.from_snes(vf, F_E, F_B));

  if (!s.mass_matrix_is_current || s.mass_matrix_every_iteration) {
    PetscCall(s.form_mass_matrix());
    s.mass_matrix_is_current = true;

    if (s.mass_matrix_check)
      PetscCall(s.check_response());
  }

  // (I + dt^2/4 rotB rotE + matL) dE = -F(E) - dt/2 rotB F(B)
  PetscCall(MatMult(s.rotB, F_B, rhs));
  PetscCall(VecScale(rhs, -0.5 * dt));
  PetscCall(VecAXPY(rhs, -1, F_E));

  KSPConvergedReason reason;
  PetscInt it;
  PetscCall(VecSet(dE, 0));
  PetscCall(KSPSolve(s.ksp, rhs, dE));
  PetscCall(KSPGetConvergedReason(s.ksp, &reason));
  PetscCall(KSPGetIterationNumber(s.ksp, &it));

  PetscCheck(reason >= 0, PETSC_COMM_WORLD, PETSC_ERR_NOT_CONVERGED,
    "Preconditioner KSPSolve has not converged, reason: %s", KSPConvergedReasons[reason]);

  LOG("Field preconditioner: KSP it={}", it);

  // dB = -F(B) - dt/2 rotE dE
  PetscCall(VecAXPY(s.E_hk, 1, dE));
  PetscCall(MatMult(s.rotE, dE, s.rE));
  PetscCall(VecAXPY(s.B_hk, -1, F_B));
  PetscCall(VecAXPY(s.B_hk, -0.5 * dt, s.rE));
  PetscCall(s.to_snes(s.E_hk, s.B_hk, vx));

  PetscCall(DMRestoreGlobalVector(s.da, &F_E));
  PetscCall(DMRestoreGlobalVector(s.da, &F_B));
  PetscCall(DMRestoreGlobalVector(s.da, &rhs));
  PetscCall(DMRestoreGlobalVector(s.da, &dE));

  PetscCall(SNESComputeFunction(npc, vx, vf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// matL = sum_p W(r_p) T_p W(r_p)^T, T_p = dt/2 dJ_p/dE_p for frozen
// trajectories, at the current positions and the node field `B_hk` of the
// particle cell:
// - guiding centers, V = v_par b + E x b / B with the mid-step v_par:
//   T = (dt/2)^2 q^2 w / m b b^T + dt/2 q w / B [. x b];
// - Crank-Nicolson, vh = R(beta) (v^n + alpha E), alpha = dt/2 q/m,
//   beta = alpha B, R w = (w + w x beta + (w, beta) beta) / (1 + beta^2):
//   T = (dt/2)^2 q^2 w / m R(beta).
PetscErrorCode Simulation::form_mass_matrix()
{
  PetscFunctionBeginUser;
  std::fill(coo_v.begin(), coo_v.end(), 0);
  PetscCall(DMGlobalToLocal(da, B_hk, INSERT_VALUES, B_loc));
  PetscCall(DMDAVecGetArrayRead(da, B_loc, &B_arr));

#pragma omp parallel for schedule(dynamic)
  for (PetscInt g = 0; g < world.size.elements_product(); g++) {
    const Vector3I vg{
      world.start[X] + g % world.size[X],
      world.start[Y] + (g / world.size[X]) % world.size[Y],
      world.start[Z] + (g / world.size[X]) / world.size[Y],
    };

    PetscReal* block = coo_v.data() + g * mm_block;
    const Vector3R& B = B_arr[vg[Z]][vg[Y]][vg[X]];
    const PetscReal lenB = B.length();

    for (const auto& sort : particles_) {
      const auto& cell = sort->get_dk_curr_storage()[g];
      if (cell.empty() || lenB == 0)
        continue;

      const PetscReal q = sort->parameters.q;
      const PetscReal m = sort->parameters.m;
      const PetscReal w = sort->parameters.n / (PetscReal)sort->parameters.Np;
      const PetscReal a = 0.25 * dt * dt * q * q * w / m;
      const PetscReal c = 0.5 * dt * q * w / lenB;
      const Vector3R b = B / lenB;

      const PetscReal T[3][3]{
        {a * b[X] * b[X], a * b[X] * b[Y] + c * b[Z], a * b[X] * b[Z] - c * b[Y]},
        {a * b[Y] * b[X] - c * b[Z], a * b[Y] * b[Y], a * b[Y] * b[Z] + c * b[X]},
        {a * b[Z] * b[X] + c * b[Y], a * b[Z] * b[Y] - c * b[X], a * b[Z] * b[Z]},
      };

      for (const auto& point : cell)
        add_to_mass_block(block, vg, point.r, T);
    }

    for (const auto& sort : kinetic_particles_) {
      const auto& cell = sort->storage[g];
      if (cell.empty())
        continue;

      const PetscReal q = sort->parameters.q;
      const PetscReal m = sort->parameters.m;
      const Vector3R beta = B * (0.5 * dt * q / m);
      const PetscReal a = 0.25 * dt * dt * q * q * sort->parameters.mpw / m / (1 + beta.squared());

      const PetscReal T[3][3]{
        {a * (1 + beta[X] * beta[X]), a * (+beta[Z] + beta[X] * beta[Y]), a * (-beta[Y] + beta[X] * beta[Z])},
        {a * (-beta[Z] + beta[Y] * beta[X]), a * (1 + beta[Y] * beta[Y]), a * (+beta[X] + beta[Y] * beta[Z])},
        {a * (+beta[Y] + beta[Z] * beta[X]), a * (-beta[X] + beta[Z] * beta[Y]), a * (1 + beta[Z] * beta[Z])},
      };

      for (const auto& point : cell)
        add_to_mass_block(block, vg, point.r, T);
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(da, B_loc, &B_arr));
  PetscCall(MatSetValuesCOO(matL, coo_v.data(), INSERT_VALUES));

  if (!matA) {
    PetscCall(MatDuplicate(matL, MAT_COPY_VALUES, &matA));
    PetscCall(MatAXPY(matA, 1, matM, DIFFERENT_NONZERO_PATTERN));
  }
  else {
    PetscCall(MatZeroEntries(matA));
    PetscCall(MatAXPY(matA, 1, matL, SUBSET_NONZERO_PATTERN));
    PetscCall(MatAXPY(matA, 1, matM, SUBSET_NONZERO_PATTERN));
  }
  PetscCall(KSPSetOperators(ksp, matA, matA));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Compares matL v with the actual response dt/2 (J(E + eps v) - J(E)) / eps
// of the pushed particles for a random v; the difference comes from the
// motion of the trajectories and the segment kernels.
PetscErrorCode Simulation::check_response()
{
  PetscFunctionBeginUser;
  Vec v, J0, Lv;
  PetscCall(DMGetGlobalVector(da, &v));
  PetscCall(DMGetGlobalVector(da, &J0));
  PetscCall(DMGetGlobalVector(da, &Lv));

  PetscRandom random;
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &random));
  PetscCall(PetscRandomSetInterval(random, -1, 1));
  PetscCall(VecSetRandom(v, random));
  PetscCall(PetscRandomDestroy(&random));

  const PetscReal eps = 1e-7;
  PetscCall(form_current());
  PetscCall(VecCopy(J, J0));
  PetscCall(VecAXPY(E_hk, eps, v));
  PetscCall(form_current());
  PetscCall(VecAXPY(E_hk, -eps, v));

  PetscCall(VecAXPY(J0, -1, J));
  PetscCall(VecScale(J0, -0.5 * dt / eps));
  PetscCall(MatMult(matL, v, Lv));

  PetscReal norm_true, norm_diff;
  PetscCall(VecNorm(J0, NORM_2, &norm_true));
  PetscCall(VecAXPY(Lv, -1, J0));
  PetscCall(VecNorm(Lv, NORM_2, &norm_diff));
  LOG("Mass matrix check: |dt/2 dJ| = {:.4e}, |matL v - dt/2 dJ| / |dt/2 dJ| = {:.4e}", norm_true, norm_diff / norm_true);

  PetscCall(DMRestoreGlobalVector(da, &v));
  PetscCall(DMRestoreGlobalVector(da, &J0));
  PetscCall(DMRestoreGlobalVector(da, &Lv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace hybrid_kinetic
