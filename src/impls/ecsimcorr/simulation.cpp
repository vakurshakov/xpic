#include "simulation.h"

#include "src/commands/builders/command_builder.h"
#include "src/diagnostics/builders/diagnostic_builder.h"
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

  for (auto& sort : particles_)
    PetscCall(sort->init());

  PetscCall(init_log_stages());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::timestep_implementation(timestep_t /* timestep */)
{
  PetscFunctionBeginUser;
  PetscLogStagePush(stagenums[0]);

  PetscCall(clear_sources());

  PetscLogStagePop();
  PetscLogStagePush(stagenums[1]);

  for (auto& sort : particles_)
    PetscCall(sort->first_push());

  PetscCall(MatAssemblyBegin(matL, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(matL, MAT_FINAL_ASSEMBLY));

  PetscLogStagePop();
  PetscLogStagePush(stagenums[2]);

  PetscCall(predict_fields());

  PetscLogStagePop();
  PetscLogStagePush(stagenums[3]);

  for (auto& sort : particles_)
    PetscCall(sort->second_push());

  PetscLogStagePop();
  PetscLogStagePush(stagenums[4]);

  PetscCall(correct_fields());

  PetscLogStagePop();
  PetscLogStagePush(stagenums[5]);

  for (auto& sort : particles_) {
    PetscCall(sort->final_update());

    /// @todo Testing petsc as a computational server first
    /// PetscCall(sort->communicate());
  }

  PetscCall(final_update());

  PetscLogStagePop();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(currI, 0.0));
  PetscCall(VecSet(currJe, 0.0));
  PetscCall(MatZeroEntries(matL));

  for (auto& sort : particles_)
    PetscCall(sort->clear_sources());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Solving Maxwell's equations to find predicted field `Ep` = E'^{n+1/2}
PetscErrorCode Simulation::predict_fields()
{
  PetscFunctionBeginUser;
  // Storing `matL` to reuse it for ECSIM current calculation later
  PetscCall(MatCopy(matL, matA, DIFFERENT_NONZERO_PATTERN));  // matA = matL
  PetscCall(MatAYPX(matA, 0.5 * POW2(dt), matM, DIFFERENT_NONZERO_PATTERN));  // matA = matM + dt^2 / 2 * matA

  PetscCall(KSPSetOperators(predict, matA, matA));
  PetscCall(KSPSetUp(predict));

  PetscCall(advance_fields(predict, currI, Ep));

  PetscCall(MatScale(matL, 0.25 * dt));  // matL *= dt / 4
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Solving Maxwell's equation to find correct `Ec` = E^{n+1/2}, satisfying continuity equation
PetscErrorCode Simulation::correct_fields()
{
  PetscFunctionBeginUser;
  PetscCall(advance_fields(correct, currJe, Ec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::advance_fields(KSP ksp, Vec curr, Vec out)
{
  PetscFunctionBeginUser;
  Vec rhs;
  PetscCall(DMGetGlobalVector(world_.da, &rhs));
  PetscCall(VecAXPY(B, -1.0, B0));

  PetscCall(VecCopy(curr, rhs));  // rhs = curr
  PetscCall(VecAXPBY(rhs, 2.0, -dt, E));  // rhs = 2 * E^{n} - (dt * rhs)
  PetscCall(MatMultAdd(rotB, B, rhs, rhs));  // rhs = rhs + rotB(B^{n})

  PetscCall(KSPSolve(ksp, rhs, out));
  PetscCall(KSPGetSolution(ksp, &out));

  PetscCall(VecAXPY(B, +1.0, B0));
  PetscCall(DMRestoreGlobalVector(world_.da, &rhs));

  // Convergence analysis
  PetscCall(KSPConvergedReasonView(ksp, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::final_update()
{
  PetscFunctionBeginUser;
  Vec util;
  PetscReal norm;
  PetscCall(DMGetGlobalVector(world_.da, &util));

  PetscCall(MatMultAdd(matL, Ec, currI, currI));  // currI = currI + matL * E^{n+1/2}
  PetscCall(VecWAXPY(util, -1, currI, currJe));  // util = -currI + currJe
  PetscCall(VecNorm(util, NORM_2, &norm));
  LOG("  Norm of the difference in ECSIM and Esirkepov currents: {:.7f}", norm);

  PetscCall(VecSet(util, 0.0));  // util = 0.0
  PetscCall(VecAXPBYPCZ(util, 2, -1, 1, Ec, E));  // E^{n+1} = 2 * E^{n+1/2} - E^{n}
  PetscCall(VecNorm(util, NORM_2, &norm));
  LOG("  Norm of the difference in electric fields between steps: {:.7f}", norm);

  PetscCall(VecSwap(util, E));
  PetscCall(DMRestoreGlobalVector(world_.da, &util));

  PetscCall(MatMultAdd(rotE, Ec, B, B));  // B^{n+1} -= dt * rot(E^{n+1/2})
  PetscFunctionReturn(PETSC_SUCCESS);
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
      return sort->parameters().sort_name == name;
    });

  if (it == particles_.end())
    throw std::runtime_error("No particles with name " + std::string(name));
  return **it;
}


PetscErrorCode Simulation::init_vectors()
{
  PetscFunctionBeginUser;
  DM da = world_.da;
  PetscCall(DMCreateGlobalVector(da, &E));
  PetscCall(DMCreateGlobalVector(da, &Ep));
  PetscCall(DMCreateGlobalVector(da, &Ec));
  PetscCall(DMCreateGlobalVector(da, &B));
  PetscCall(DMCreateGlobalVector(da, &B0));
  PetscCall(DMCreateGlobalVector(da, &currI));
  PetscCall(DMCreateGlobalVector(da, &currJe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_matrices()
{
  PetscFunctionBeginUser;
  DM da = world_.da;
  PetscCall(DMCreateMatrix(da, &matL));
  PetscCall(MatSetOption(matL, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));
  PetscCall(MatDuplicate(matL, MAT_DO_NOT_COPY_VALUES, &matA));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rotE));
  PetscCall(rotor.create_negative(&rotB));

  PetscCall(MatProductCreate(rotB, rotE, nullptr, &matM));
  PetscCall(MatProductSetType(matM, MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(matM));
  PetscCall(MatProductSymbolic(matM));
  PetscCall(MatProductNumeric(matM));  // matM = rotB(rotE())
  PetscCall(MatScale(matM, 0.5 * POW2(dt)));  // matM = dt^2 / 2 * matM
  PetscCall(MatShift(matM, 2.0));  // matM = 2 * I + matM

  PetscCall(MatScale(rotE, -dt));
  PetscCall(MatScale(rotB, +dt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_ksp_solvers()
{
  PetscFunctionBeginUser;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &predict));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &correct));
  PetscCall(KSPSetErrorIfNotConverged(predict, PETSC_TRUE));
  PetscCall(KSPSetErrorIfNotConverged(correct, PETSC_TRUE));

  static constexpr PetscReal atol = 1e-10;
  static constexpr PetscReal rtol = 1e-10;
  PetscCall(KSPSetTolerances(predict, atol, rtol, PETSC_CURRENT, PETSC_CURRENT));
  PetscCall(KSPSetTolerances(correct, atol, rtol, PETSC_CURRENT, PETSC_CURRENT));

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
  PetscCallVoid(VecDestroy(&E));
  PetscCallVoid(VecDestroy(&Ep));
  PetscCallVoid(VecDestroy(&Ec));
  PetscCallVoid(VecDestroy(&B));
  PetscCallVoid(VecDestroy(&B0));
  PetscCallVoid(VecDestroy(&currI));
  PetscCallVoid(VecDestroy(&currJe));

  PetscCallVoid(MatDestroy(&matL));
  PetscCallVoid(MatDestroy(&matA));
  PetscCallVoid(MatDestroy(&matM));
  PetscCallVoid(MatDestroy(&rotE));
  PetscCallVoid(MatDestroy(&rotB));

  PetscCallVoid(KSPDestroy(&predict));
  PetscCallVoid(KSPDestroy(&correct));
  PetscFunctionReturnVoid();
}

}  // namespace ecsimcorr
