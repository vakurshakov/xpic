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
  PetscCall(init_log_stages());
  PetscCall(clock.push(stagenums[0]));
  PetscCall(PetscLogStagePush(stagenums[0]));

  assembly_map.resize(world.size.elements_product());

  PetscCall(init_vectors());
  PetscCall(init_matrices());
  PetscCall(init_ksp_solvers());
  PetscCall(init_particles());

  std::vector<Command_up> presets;
  PetscCall(build_commands(*this, "Presets", presets));
  PetscCall(build_commands(*this, "StepPresets", step_presets_));

  LOG("Executing presets");
  for (auto&& preset : presets)
    preset->execute(0);

  PetscCall(VecAXPY(B, 1.0, B0));

  PetscCall(build_diagnostics(*this, diagnostics_));
  diagnostics_.emplace_back(std::make_unique<EnergyConservation>(*this));
  diagnostics_.emplace_back(std::make_unique<ChargeConservation>(*this));
  diagnostics_.emplace_back(std::make_unique<MatDump>(CONFIG().out_dir + "/matL",
    matL, "./results/energy-test/mirror/assembly_control_mpi_1/matL/"));

  for (auto& sort : particles_)
    PetscCall(sort->calculate_energy());

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());

  LOG("Initialization took {:6.4e} seconds", clock.get(stagenums[0]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::timestep_implementation(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  PetscCall(clear_sources());
  PetscCall(first_push());
  PetscCall(predict_fields());
  PetscCall(second_push());
  PetscCall(correct_fields());
  PetscCall(final_update());
  PetscCall(log_timings());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(stagenums[1]));
  PetscCall(PetscLogStagePush(stagenums[1]));

  PetscCall(VecSet(currI, 0.0));
  PetscCall(VecSet(currJe, 0.0));
  PetscCall(MatZeroEntries(matL));

  for (auto& sort : particles_)
    PetscCall(sort->clear_sources());

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::first_push()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(stagenums[2]));
  PetscCall(PetscLogStagePush(stagenums[2]));

  for (auto& sort : particles_)
    PetscCall(sort->first_push());

  PetscCall(update_cells());
  PetscCall(fill_ecsim_current());

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Solving Maxwell's equations to find predicted field `Ep` = E'^{n+1/2}
PetscErrorCode Simulation::predict_fields()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(stagenums[3]));
  PetscCall(PetscLogStagePush(stagenums[3]));

  // Storing `matL` to reuse it for ECSIM current calculation later
  Mat matA;
  PetscCall(MatDuplicate(matL, MAT_COPY_VALUES, &matA));  // matA = matL
  PetscCall(MatAYPX(matA, 2.0 * dt, matM, DIFFERENT_NONZERO_PATTERN));  // matA = matM + (2 * dt) * matA

  // Note that we use `matM` to construct the preconditioning matrix
  PetscCall(KSPSetOperators(predict, matA, matM));
  PetscCall(KSPSetUp(predict));

  PetscCall(advance_fields(predict, currI, Ep));
  PetscCall(MatDestroy(&matA));

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::second_push()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(stagenums[4]));
  PetscCall(PetscLogStagePush(stagenums[4]));

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

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Solving Maxwell's equation to find correct `Ec` = E^{n+1/2}, satisfying continuity equation
PetscErrorCode Simulation::correct_fields()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(stagenums[5]));
  PetscCall(PetscLogStagePush(stagenums[5]));

  PetscCall(advance_fields(correct, currJe, Ec));

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::final_update()
{
  PetscFunctionBeginUser;
  PetscCall(clock.push(stagenums[6]));
  PetscCall(PetscLogStagePush(stagenums[6]));

  for (auto& sort : particles_)
    PetscCall(sort->final_update());

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

  PetscCall(PetscLogStagePop());
  PetscCall(clock.pop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::log_timings()
{
  PetscFunctionBeginUser;
  PetscInt size = (PetscInt)std::size(stagenums);
  PetscLogDouble sum = 0.0;

  /// @note We are skipping the initialization stage
  for (PetscInt i = 1; i < size; ++i)
    sum += clock.get(stagenums[i]);

  LOG("  Summary of Stages:  ------- Time -------");
  LOG("                          Avg         %");

  for (PetscInt i = 1; i < size; ++i) {
    PetscLogStage id = stagenums[i];

    const char* name;
    PetscCall(PetscLogStageGetName(id, &name));

    PetscLogDouble time = clock.get(std::string(name));
    LOG("  {:2d}: {:>15s}: {:6.4e}  {:5.1f}%", id, name, time, 100.0 * time / sum);
  }
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
/// `indices_assembled` should include particles of all sorts.
/// @note This routine _must_ be called before `fill_matrix_indices()`
PetscErrorCode Simulation::update_cells()
{
  PetscFunctionBeginUser;
#if UPDATE_CELLS_SEQ
  for (auto& sort : particles_)
    PetscCall(sort->update_cells());
#else
  for (auto& sort : particles_)
    PetscCall(sort->update_cells_mpi());
#endif

  static std::vector<bool> prev_assembly_map;
  prev_assembly_map = assembly_map;

  for (const auto& sort : particles_) {
    for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
      if (sort->storage[g].empty() || prev_assembly_map[g])
        continue;

      if (indices_assembled) {
        LOG("  Indices assembly has been broken by \"{}\"", sort->parameters.sort_name);
      }
      indices_assembled = false;

      Vector3I vg{
        world.start[X] + g % world.size[X],
        world.start[Y] + (g / world.size[X]) % world.size[Y],
        world.start[Z] + (g / world.size[X]) / world.size[Y],
      };

      for (PetscInt i = 0; i < POW3(assembly_width); ++i) {
        Vector3I vgi{
          vg[X] - assembly_radius + i % assembly_width,
          vg[Y] - assembly_radius + (i / assembly_width) % assembly_width,
          vg[Z] - assembly_radius + (i / assembly_width) / assembly_width,
        };

        if (!is_point_within_bounds(vgi, world.start, world.size))
          continue;

        PetscInt j = world.s_g(  //
          vgi[X] - world.start[X],  //
          vgi[Y] - world.start[Y],  //
          vgi[Z] - world.start[Z]);

        assembly_map[j] = true;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::fill_ecsim_current()
{
  PetscFunctionBeginUser;
  PetscInt size = 0;
  get_array_offset(0, world.size.elements_product(), size);

  static std::vector<MatStencil> coo_i;
  static std::vector<MatStencil> coo_j;
  static std::vector<PetscReal> coo_v;

  if (!indices_assembled) {
    PetscReal ind_mem, val_mem;
    ind_mem = (PetscReal)size * 2 * sizeof(MatStencil) / 1e9;
    val_mem = (PetscReal)size * sizeof(PetscReal) / 1e9;
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &ind_mem, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &val_mem, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));

    LOG("  Indices assembly was broken, recreating them");
    LOG("  Additional space for indices: {:4.3f} GB, values: {:4.3f} GB", ind_mem, val_mem);

    coo_i.resize(size);
    coo_j.resize(size);
    coo_v.resize(size);

    PetscCall(fill_matrix_indices(coo_i.data(), coo_j.data()));
    PetscCall(mat_set_preallocation_coo(size, coo_i.data(), coo_j.data()));
    indices_assembled = true;
  }

  std::fill(coo_v.begin(), coo_v.end(), 0.0);
  PetscCall(fill_ecsim_current(coo_v.data()));

  PetscCall(MatSetValuesCOO(matL, coo_v.data(), ADD_VALUES));
  PetscCall(MatScale(matL, 0.25 * dt));  // matL *= dt / 4
  PetscFunctionReturn(PETSC_SUCCESS);
}

constexpr PetscInt ind(PetscInt g, PetscInt c1, PetscInt c2)
{
  return g * POW2(3) + (c1 * 3 + c2);
}

/// @note Indices are assembled on those cells `g`, where `assembly_map[g] == true`
PetscErrorCode Simulation::fill_matrix_indices(
  MatStencil* coo_i, MatStencil* coo_j)
{
  PetscFunctionBeginUser;
  constexpr PetscInt shr = 1;
  constexpr PetscInt shw = 2 * shr + 1;

  PetscInt prev_g = 0;
  PetscInt off = 0;

#pragma omp parallel for firstprivate(prev_g, off)
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    if (!assembly_map[g])
      continue;

    get_array_offset(prev_g, g, off);
    prev_g = g;

    MatStencil* coo_ci = coo_i + off;
    MatStencil* coo_cj = coo_j + off;

    Vector3I vg{
      world.start[X] + g % world.size[X],
      world.start[Y] + (g / world.size[X]) % world.size[Y],
      world.start[Z] + (g / world.size[X]) / world.size[Y],
    };

    for (PetscInt g1 = 0; g1 < POW3(shw); ++g1) {
      for (PetscInt g2 = 0; g2 < POW3(shw); ++g2) {
        PetscInt gg = g1 * POW3(shw) + g2;

        Vector3I vg1{
          vg[X] + g1 % shw - shr,
          vg[Y] + (g1 / shw) % shw - shr,
          vg[Z] + (g1 / shw) / shw - shr,
        };

        Vector3I vg2{
          vg[X] + g2 % shw - shr,
          vg[Y] + (g2 / shw) % shw - shr,
          vg[Z] + (g2 / shw) / shw - shr,
        };

        for (PetscInt c = 0; c < Vector3I::dim; ++c) {
          coo_ci[ind(gg, X, c)] = MatStencil{vg1[Z], vg1[Y], vg1[X], X};
          coo_ci[ind(gg, Y, c)] = MatStencil{vg1[Z], vg1[Y], vg1[X], Y};
          coo_ci[ind(gg, Z, c)] = MatStencil{vg1[Z], vg1[Y], vg1[X], Z};

          coo_cj[ind(gg, c, X)] = MatStencil{vg2[Z], vg2[Y], vg2[X], X};
          coo_cj[ind(gg, c, Y)] = MatStencil{vg2[Z], vg2[Y], vg2[X], Y};
          coo_cj[ind(gg, c, Z)] = MatStencil{vg2[Z], vg2[Y], vg2[X], Z};
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::fill_ecsim_current(PetscReal* coo_v)
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

  PetscLogEventBegin(events[0], local_B, 0, 0, 0);

  for (const auto& sort : particles_) {
    PetscInt prev_g = 0;
    PetscInt off = 0;

#pragma omp parallel for firstprivate(prev_g, off)
    for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
      if (sort->storage[g].empty())
        continue;

      get_array_offset(prev_g, g, off);
      prev_g = g;

      PetscReal* coo_cv = coo_v + off;
      sort->fill_ecsim_current(g, coo_cv);
    }
  }

  PetscLogEventEnd(events[0], local_B, 0, 0, 0);

  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &arr_B));

  for (auto& sort : particles_) {
    PetscCall(DMDAVecRestoreArrayWrite(da, sort->local_currI, &sort->currI));
    PetscCall(DMLocalToGlobal(da, sort->local_currI, ADD_VALUES, sort->global_currI));
    PetscCall(VecAXPY(currI, 1.0, sort->global_currI));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @returns The proper offset of cell `g` into a global arrays
void Simulation::get_array_offset(PetscInt begin_g, PetscInt end_g, PetscInt& off)
{
  constexpr PetscInt shs = POW2(3 * POW3(3));
  for (PetscInt g = begin_g; g < end_g; ++g)
    if (assembly_map[g])
      off += shs;
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
  const std::map<std::string, KSP&> map{
    {"predict", predict},
    {"correct", correct},
  };

  static constexpr PetscReal atol = 1e-10;
  static constexpr PetscReal rtol = 1e-10;

  for (auto&& [name, ksp] : map) {
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(PetscObjectSetName((PetscObject)ksp, name.c_str()));
    PetscCall(KSPSetOptionsPrefix(ksp, (name + "_").c_str()));

    PetscCall(KSPSetTolerances(ksp, atol, rtol, PETSC_CURRENT, PETSC_CURRENT));
    PetscCall(KSPSetErrorIfNotConverged(ksp, PETSC_TRUE));
    PetscCall(KSPSetReusePreconditioner(ksp, PETSC_TRUE));
    PetscCall(KSPSetFromOptions(ksp));
  }

  PetscCall(KSPSetOperators(correct, matM, matM));
  PetscCall(KSPSetUp(correct));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_particles()
{
  PetscFunctionBeginUser;
  const Configuration::json_t& particles_info = CONFIG().json.at("Particles");
  for (auto&& info : particles_info) {
    SortParameters parameters;
    info.at("sort_name").get_to(parameters.sort_name);
    info.at("Np").get_to(parameters.Np);
    info.at("n").get_to(parameters.n);
    info.at("q").get_to(parameters.q);
    info.at("m").get_to(parameters.m);

    if (info.contains("T")) {
      PetscReal T;
      info.at("T").get_to(T);
      parameters.Tx = T;
      parameters.Ty = T;
      parameters.Tz = T;
    }
    else {
      info.at("Tx").get_to(parameters.Tx);
      info.at("Ty").get_to(parameters.Ty);
      info.at("Tz").get_to(parameters.Tz);
    }

    particles_.emplace_back(std::make_unique<Particles>(*this, parameters));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_log_stages()
{
  PetscFunctionBeginUser;
  PetscCall(PetscClassIdRegister("ecsimcorr::Simulation", &classid));
  PetscCall(PetscLogEventRegister("fill_ecsim_curr", classid, &events[0]));

  PetscCall(PetscLogStageRegister("Initialization", &stagenums[0]));
  PetscCall(PetscLogStageRegister("Clear sources", &stagenums[1]));
  PetscCall(PetscLogStageRegister("First push", &stagenums[2]));
  PetscCall(PetscLogStageRegister("Predict field", &stagenums[3]));
  PetscCall(PetscLogStageRegister("Second push", &stagenums[4]));
  PetscCall(PetscLogStageRegister("Correct fields", &stagenums[5]));
  PetscCall(PetscLogStageRegister("Renormalization", &stagenums[6]));
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
