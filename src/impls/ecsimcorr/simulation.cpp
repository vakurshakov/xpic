#include "simulation.h"

#include "src/commands/builders/command_builder.h"
#include "src/diagnostics/builders/diagnostic_builder.h"
#include "src/diagnostics/mat_dump.h"
#include "src/impls/ecsimcorr/charge_conservation.h"
#include "src/impls/ecsimcorr/energy_conservation.h"
#include "src/utils/operators.h"
#include "src/utils/particles_load.hpp"
#include "src/utils/random_generator.h"
#include "src/utils/utils.h"


namespace ecsimcorr {

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  PetscCall(init_vectors());
  PetscCall(init_matrices());
  PetscCall(init_ksp_solvers());

  const Configuration::json_t& particles_info = CONFIG().json.at("Particles");
  for (auto&& info : particles_info) {
    SortParameters parameters;
    info.at("sort_name").get_to(parameters.sort_name);
    info.at("Np").get_to(parameters.Np);
    info.at("n").get_to(parameters.n);
    info.at("q").get_to(parameters.q);
    info.at("m").get_to(parameters.m);
    info.at("T").get_to(parameters.Tx);
    info.at("T").get_to(parameters.Ty);
    info.at("T").get_to(parameters.Tz);
    particles_.emplace_back(std::make_unique<Particles>(*this, parameters));
  }

  std::list<Command_up> presets;
  PetscCall(build_commands(*this, "Presets", presets));
  PetscCall(build_commands(*this, "StepPresets", step_presets_));

  LOG("Executing presets");
  for (auto&& preset : presets)
    preset->execute(0);

  PetscCall(VecAXPY(B, 1.0, B0));

  PetscCall(build_diagnostics(*this, diagnostics_));
  diagnostics_.emplace_back(std::make_unique<EnergyConservation>(*this));
  diagnostics_.emplace_back(std::make_unique<ChargeConservation>(*this));
  diagnostics_.emplace_back(
    std::make_unique<MatDump>(CONFIG().out_dir + "/ecsim/matL/", matL,
      "./results/performance-test/mpi-parallelization/"
      "MatSetValuesBlockedStencil/ecsim/matL"));

  for (auto& sort : particles_)
    PetscCall(sort->calculate_energy());

  PetscCall(init_log_stages());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::timestep_implementation(timestep_t /* t */)
{
  PetscFunctionBeginUser;
  PetscCall(clear_sources());
  PetscCall(first_push());
  PetscCall(predict_fields());
  PetscCall(second_push());
  PetscCall(correct_fields());
  PetscCall(final_update());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::clear_sources()
{
  PetscFunctionBeginUser;
  PetscLogStagePush(stagenums[0]);

  PetscCall(VecSet(currI, 0.0));
  PetscCall(VecSet(currJe, 0.0));
  PetscCall(MatZeroEntries(matL));

  for (auto& sort : particles_)
    PetscCall(sort->clear_sources());

  PetscLogStagePop();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::first_push()
{
  PetscFunctionBeginUser;
  PetscLogStagePush(stagenums[1]);

  for (auto& sort : particles_) {
    PetscCall(sort->first_push());
    PetscCall(sort->correct_coordinates());
  }

  PetscCall(update_cells());
  PetscCall(fill_ecsim_current());

  PetscLogStagePop();
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Solving Maxwell's equations to find predicted field `Ep` = E'^{n+1/2}
PetscErrorCode Simulation::predict_fields()
{
  PetscFunctionBeginUser;
  PetscLogStagePush(stagenums[2]);

  // Storing `matL` to reuse it for ECSIM current calculation later
  Mat matA;
  PetscCall(MatDuplicate(matL, MAT_COPY_VALUES, &matA));  // matA = matL
  PetscCall(MatAYPX(matA, 2.0 * dt, matM, DIFFERENT_NONZERO_PATTERN));  // matA = matM + (2 * dt) * matA

  // Note that we use `matM` to construct the preconditioning matrix
  PetscCall(KSPSetOperators(predict, matA, matM));
  PetscCall(KSPSetUp(predict));

  PetscCall(advance_fields(predict, currI, Ep));
  PetscCall(MatDestroy(&matA));

  PetscLogStagePop();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::second_push()
{
  PetscFunctionBeginUser;
  PetscLogStagePush(stagenums[3]);

  DM da = world.da;
  PetscCall(DMGlobalToLocal(da, Ep, INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, B, INSERT_VALUES, local_B));

  Vector3R*** arr_E;
  Vector3R*** arr_B;
  PetscCall(DMDAVecGetArrayRead(da, local_E, &arr_E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &arr_B));

  for (auto& sort : particles_) {
    sort->E = arr_E;
    sort->B = arr_B;
    PetscCall(sort->second_push());
    PetscCall(sort->correct_coordinates());
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &arr_E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &arr_B));

  PetscCall(update_cells());

  PetscLogStagePop();
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Solving Maxwell's equation to find correct `Ec` = E^{n+1/2}, satisfying continuity equation
PetscErrorCode Simulation::correct_fields()
{
  PetscFunctionBeginUser;
  PetscLogStagePush(stagenums[4]);

  PetscCall(advance_fields(correct, currJe, Ec));

  PetscLogStagePop();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::final_update()
{
  PetscFunctionBeginUser;
  PetscLogStagePush(stagenums[5]);

  for (auto& sort : particles_) {
    PetscCall(sort->final_update());

    /// @todo Testing petsc as a computational server first
    /// PetscCall(sort->communicate());
  }

  Vec util;
  PetscReal norm;
  PetscCall(DMGetGlobalVector(world.da, &util));

  PetscCall(MatMultAdd(matL, Ec, currI, currI));  // currI = currI + matL * E^{n+1/2}
  PetscCall(VecWAXPY(util, -1, currI, currJe));  // util = -currI + currJe
  PetscCall(VecNorm(util, NORM_2, &norm));
  LOG("  Norm of the difference in ECSIM and Esirkepov currents: {:.7f}", norm);

  PetscCall(VecSet(util, 0.0));  // util = 0.0
  PetscCall(VecAXPBYPCZ(util, 2, -1, 1, Ec, E));  // E^{n+1} = 2 * E^{n+1/2} - E^{n}
  PetscCall(VecNorm(util, NORM_2, &norm));
  LOG("  Norm of the difference in electric fields between steps: {:.7f}", norm);

  PetscCall(VecSwap(util, E));
  PetscCall(DMRestoreGlobalVector(world.da, &util));

  PetscCall(MatMultAdd(rotE, Ec, B, B));  // B^{n+1} -= dt * rot(E^{n+1/2})

  PetscLogStagePop();
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::advance_fields(KSP ksp, Vec curr, Vec out)
{
  PetscFunctionBeginUser;
  Vec rhs;
  PetscCall(DMGetGlobalVector(world.da, &rhs));
  PetscCall(VecAXPY(B, -1.0, B0));

  PetscCall(VecCopy(curr, rhs));  // rhs = curr
  PetscCall(VecAXPBY(rhs, 2.0, -dt, E));  // rhs = 2 * E^{n} - (dt * rhs)
  PetscCall(MatMultAdd(rotB, B, rhs, rhs));  // rhs = rhs + rotB(B^{n})

  PetscCall(KSPSolve(ksp, rhs, out));
  PetscCall(KSPGetSolution(ksp, &out));

  PetscCall(VecAXPY(B, +1.0, B0));
  PetscCall(DMRestoreGlobalVector(world.da, &rhs));

  // Convergence analysis
  const char* name;
  PetscCall(PetscObjectGetName((PetscObject)ksp, &name));
  LOG("  KSPSolve() has finished for \"{}\", KSPConvergedReasonView():", name);
  PetscCall(KSPConvergedReasonView(ksp, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @note Since we use one global Lapenta matrix, test on
/// `matL_indices_assembled` should include particles of all sorts.
/// @note This routine _must_ be called before `fill_ecsim_current()`
PetscErrorCode Simulation::update_cells()
{
  PetscFunctionBeginUser;
  for (auto& sort : particles_) {
    for (PetscInt g = 0; g < geom_nz * geom_ny * geom_nx; ++g) {
      auto& storage_g = sort->storage[g];
      auto it = storage_g.begin();
      while (it != storage_g.end()) {
        auto ng = indexing::s_g(  //
          static_cast<PetscInt>(it->z() / dz),  //
          static_cast<PetscInt>(it->y() / dy),  //
          static_cast<PetscInt>(it->x() / dx));

        if (ng == g) {
          it = std::next(it);
          continue;
        }

        for (const auto& check_sort : particles_) {
          if (!matL_indices_assembled || !check_sort->storage[ng].empty())
            continue;

          LOG("  Indices assembly is broken by \"{}\"", sort->parameters.sort_name);
          matL_indices_assembled = false;
        }

        auto& storage_ng = sort->storage[ng];
        storage_ng.emplace_back(*it);
        it = storage_g.erase(it);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::fill_ecsim_current()
{
  PetscFunctionBeginUser;
  PetscInt size = 0;
  get_array_offset(0, geom_nz * geom_ny * geom_nx, size);

  std::vector<MatStencil> coo_i;
  std::vector<MatStencil> coo_j;
  std::vector<PetscReal> coo_v;

  if (!matL_indices_assembled) {
    coo_i.resize(size);
    coo_j.resize(size);
    LOG("  Indices assembly was broken, recollecting them again."
        " Additional space: {:4.3f} GB", (PetscReal)size * 2 * 4 * sizeof(PetscInt) / 1e9);
  }

  coo_v.resize(size, 0.0);
  LOG("  To collect matrix values, temporary storage of size {:4.3f} GB was allocated",
      (PetscReal)size * sizeof(PetscReal) / 1e9);

  fill_ecsim_current(coo_i.data(), coo_j.data(), coo_v.data());

  if (!matL_indices_assembled) {
    PetscCall(mat_set_preallocation_coo(size, coo_i.data(), coo_j.data()));
    matL_indices_assembled = true;
  }

  PetscCall(MatSetValuesCOO(matL, coo_v.data(), ADD_VALUES));
  PetscCall(MatScale(matL, 0.25 * dt));  // matL *= dt / 4
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::fill_ecsim_current(
  MatStencil* coo_i, MatStencil* coo_j, PetscReal* coo_v)
{
  PetscFunctionBeginUser;
  Vector3R*** arr_B;

  DM da = world.da;
  PetscCall(DMGlobalToLocal(da, B, INSERT_VALUES, local_B));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &arr_B));

  for (auto& sort : particles_) {
    sort->B = arr_B;
    PetscCall(DMDAVecGetArrayWrite(da, sort->local_currI, &sort->currI));
  }

  static constexpr PetscInt OMP_CHUNK_SIZE = 16;
  PetscInt prev_g = 0;
  PetscInt off = 0;

#pragma omp parallel for firstprivate(prev_g, off) \
  schedule(monotonic : dynamic, OMP_CHUNK_SIZE)
  for (PetscInt g = 0; g < geom_nz * geom_ny * geom_nx; ++g) {
    for (const auto& sort : particles_) {
      if (sort->storage[g].empty())
        continue;

      get_array_offset(prev_g, g, off);
      prev_g = g;

      if (!matL_indices_assembled) {
        MatStencil* coo_ci = coo_i + off;
        MatStencil* coo_cj = coo_j + off;
        sort->fill_matrix_indices(g, coo_ci, coo_cj);
      }

      PetscReal* coo_cv = coo_v + off;
      sort->fill_ecsim_current(g, coo_cv);
    }
  }

  for (auto& sort : particles_) {
    PetscCall(DMDAVecRestoreArrayWrite(da, sort->local_currI, &sort->currI));
    PetscCall(DMLocalToGlobal(da, sort->local_currI, ADD_VALUES, sort->global_currI));
    PetscCall(VecAXPY(currI, 1.0, sort->global_currI));
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &arr_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @returns The proper offset of cell `g` this into a global arrays
void Simulation::get_array_offset(PetscInt begin_g, PetscInt end_g, PetscInt& off)
{
  constexpr PetscInt shs = POW2(3 * POW3(3));
  for (PetscInt i = begin_g; i < end_g; ++i) {
    for (const auto& sort : particles_) {
      if (!sort->storage[i].empty()) {
        off += shs;
        break;
      }
    }
  }
}

PetscErrorCode Simulation::mat_set_preallocation_coo(
  PetscInt size, MatStencil* coo_i, MatStencil* coo_j)
{
  PetscFunctionBeginUser;
  auto idxm = (PetscInt*)coo_i;
  auto idxn = (PetscInt*)coo_j;

  DM da = world.da;
  Operator::remap_stencil(da, 3, size, idxm);
  Operator::remap_stencil(da, 3, size, idxn);

  PetscCall(MatSetPreallocationCOOLocal(matL, size, idxm, idxn));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::init_vectors()
{
  PetscFunctionBeginUser;
  DM da = world.da;
  PetscCall(DMCreateGlobalVector(da, &E));
  PetscCall(DMCreateGlobalVector(da, &Ep));
  PetscCall(DMCreateGlobalVector(da, &Ec));
  PetscCall(DMCreateGlobalVector(da, &B));
  PetscCall(DMCreateGlobalVector(da, &B0));
  PetscCall(DMCreateGlobalVector(da, &currI));
  PetscCall(DMCreateGlobalVector(da, &currJe));

  PetscCall(DMCreateLocalVector(da, &local_E));
  PetscCall(DMCreateLocalVector(da, &local_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_matrices()
{
  PetscFunctionBeginUser;
  DM da = world.da;
  PetscCall(DMSetMatrixPreallocateOnly(da, PETSC_FALSE));
  PetscCall(DMSetMatrixPreallocateSkip(da, PETSC_TRUE));

  PetscCall(DMCreateMatrix(da, &matL));
  PetscCall(MatSetOption(matL, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rotE));
  PetscCall(rotor.create_negative(&rotB));

  RotorMult rotor_mult(da);
  PetscCall(rotor_mult.create(&matM));  // matM = rotB(rotE())
  PetscCall(MatScale(matM, 0.5 * POW2(dt)));  // matM = dt^2 / 2 * matM
  PetscCall(MatShift(matM, 2.0));  // matM = 2 * I + matM

  PetscCall(MatScale(rotE, -dt));
  PetscCall(MatScale(rotB, +dt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Both `predict` and `correct` use the same constant `matM` as a preconditioner
PetscErrorCode Simulation::init_ksp_solvers()
{
  PetscFunctionBeginUser;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &predict));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &correct));
  PetscCall(PetscObjectSetName((PetscObject)predict, "predict"));
  PetscCall(PetscObjectSetName((PetscObject)correct, "correct"));
  PetscCall(KSPSetOptionsPrefix(predict, "predict_"));
  PetscCall(KSPSetOptionsPrefix(correct, "correct_"));

  static constexpr PetscReal atol = 1e-10;
  static constexpr PetscReal rtol = 1e-10;
  PetscCall(KSPSetTolerances(predict, atol, rtol, PETSC_CURRENT, PETSC_CURRENT));
  PetscCall(KSPSetTolerances(correct, atol, rtol, PETSC_CURRENT, PETSC_CURRENT));
  PetscCall(KSPSetErrorIfNotConverged(predict, PETSC_TRUE));
  PetscCall(KSPSetErrorIfNotConverged(correct, PETSC_TRUE));
  PetscCall(KSPSetReusePreconditioner(predict, PETSC_TRUE));
  PetscCall(KSPSetReusePreconditioner(correct, PETSC_TRUE));

  PetscCall(KSPSetOperators(correct, matM, matM));
  PetscCall(KSPSetUp(correct));

  PetscCall(KSPSetFromOptions(predict));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_log_stages()
{
  PetscFunctionBeginUser;
  PetscLogStageRegister("Clear sources", &stagenums[0]);
  PetscLogStageRegister("First push", &stagenums[1]);
  PetscLogStageRegister("Predict field", &stagenums[2]);
  PetscLogStageRegister("Second push", &stagenums[3]);
  PetscLogStageRegister("Correct fields", &stagenums[4]);
  PetscLogStageRegister("Renormalization", &stagenums[5]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

Simulation::~Simulation()
{
  PetscFunctionBeginUser;
  PetscCallVoid(KSPDestroy(&predict));
  PetscCallVoid(KSPDestroy(&correct));

  PetscCallVoid(MatDestroy(&matL));
  PetscCallVoid(MatDestroy(&matM));
  PetscCallVoid(MatDestroy(&rotE));
  PetscCallVoid(MatDestroy(&rotB));

  PetscCallVoid(VecDestroy(&E));
  PetscCallVoid(VecDestroy(&Ep));
  PetscCallVoid(VecDestroy(&Ec));
  PetscCallVoid(VecDestroy(&B));
  PetscCallVoid(VecDestroy(&B0));
  PetscCallVoid(VecDestroy(&currI));
  PetscCallVoid(VecDestroy(&currJe));

  PetscCallVoid(VecDestroy(&local_E));
  PetscCallVoid(VecDestroy(&local_B));
  PetscFunctionReturnVoid();
}


Vec Simulation::get_named_vector(std::string_view name)
{
  if (name == "E")
    return E;
  if (name == "E_pred")
    return Ep;
  if (name == "E_corr")
    return Ec;
  if (name == "B")
    return B;
  if (name == "B0")
    return B0;
  if (name == "J_ecsim")
    return currI;
  if (name == "J_esirkepov")
    return currJe;
  throw std::runtime_error("Unknown vector name " + std::string(name));
}

Particles& Simulation::get_named_particles(std::string_view name)
{
  auto it = std::find_if(particles_.begin(), particles_.end(),  //
    [&](const auto& sort) {
      return sort->parameters.sort_name == name;
    });

  if (it == particles_.end())
    throw std::runtime_error("No particles with name " + std::string(name));
  return **it;
}

}  // namespace ecsimcorr
