#include "simulation.h"

#include "src/commands/builders/command_builder.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"


namespace ecsimcorr {

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  PetscCall(init_vectors());
  PetscCall(init_matrices());
  PetscCall(init_ksp_solvers());

  PetscCall(build_commands(*this, "StepPresets", step_presets_));

  std::list<Command_up> presets;
  PetscCall(build_commands(*this, "Presets", presets));

  for (const Command_up& command : presets)
    PetscCall(command->execute(0));

  SortParameters parameters{
    .Np = 1,
    .n = +1.0,
    .q = -1.0,
    .m = +1.0,
    .sort_name = "electrons",
  };
  auto& sort = particles_.emplace_back(*this, parameters);
  sort.add_particle(Point{{geom_x / 2, geom_y / 2, geom_z / 4}, {0.0, 0.0, 0.5}});
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_vectors()
{
  PetscFunctionBeginUser;
  DM da = world_.da;
  PetscCall(DMCreateGlobalVector(da, &E));
  PetscCall(DMCreateGlobalVector(da, &En));
  PetscCall(DMCreateGlobalVector(da, &B));
  PetscCall(DMCreateGlobalVector(da, &B0));
  PetscCall(DMCreateGlobalVector(da, &currI));
  PetscCall(DMCreateGlobalVector(da, &currJ));
  PetscCall(DMCreateGlobalVector(da, &currJe));

  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(E),  "E^n"));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(En), "E^{n+1/2}"));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(B),  "B^n"));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(B0), "B^0"));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(currI), "I"));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(currI), "J_{ecsim}"));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(currI), "J_{esirkepov}"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::init_matrices()
{
  PetscFunctionBeginUser;
  DM da = world_.da;
  PetscCall(DMCreateMatrix(da, &matL));
  PetscCall(MatDuplicate(matL, MAT_DO_NOT_COPY_VALUES, &matA));
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(matL), "Lapenta matrix"));

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
  PetscCall(PetscObjectSetName(reinterpret_cast<PetscObject>(matM), "Utility matrix"));

  PetscCall(MatScale(rotE, -dt));
  PetscCall(MatScale(rotB, +dt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @todo We can optimize `correct` ksp further, by choosing appropriate KSPType
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

  PC pc;
  PetscCall(KSPGetPC(predict, &pc));

  // `matA` can be overwritten during `advance_fields()`, we would use only stored copy of `matL`
  PetscCall(PCFactorSetUseInPlace(pc, PETSC_TRUE));

  PetscCall(KSPSetOperators(correct, matM, matM));
  PetscCall(KSPSetUp(correct));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::timestep_implementation(timestep_t /* timestep */)
{
  PetscFunctionBeginUser;
  PetscCall(clear_sources());

  for (auto& sort : particles_)
    PetscCall(sort.first_push());

  PetscCall(predict_fields());

  for (auto& sort : particles_)
    PetscCall(sort.second_push());

  PetscCall(correct_fields());

  for (auto& sort : particles_) {
    PetscCall(sort.final_update());
    PetscCall(sort.communicate());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::clear_sources()
{
  PetscFunctionBeginUser;
  w1 = 0.0;
  w2 = 0.0;
  PetscCall(VecSet(currI, 0.0));
  PetscCall(VecSet(currJ, 0.0));
  PetscCall(VecSet(currJe, 0.0));
  PetscCall(MatZeroEntries(matL));

  for (auto& sort : particles_)
    PetscCall(sort.clear_sources());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::predict_fields()
{
  PetscFunctionBeginUser;
  // Storing identity current `currI` before we use it
  // as a storage for the right hand side of the `ksp`
  PetscCall(VecCopy(currI, currJ));  // currJ = currI

  // The same copying is made for `matL`
  PetscCall(MatCopy(matL, matA, DIFFERENT_NONZERO_PATTERN));  // matA = matL
  PetscCall(MatAYPX(matA, 0.5 * POW2(dt), matM, DIFFERENT_NONZERO_PATTERN));  // matA = matM + dt^2 / 2 * matA

  PetscCall(KSPSetOperators(predict, matA, matA));
  PetscCall(KSPSetUp(predict));

  // Solving the Maxwell's equations to find prediction of E'^{n+1/2}
  PetscCall(advance_fields(predict, currI));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @todo Do we use successive solve here? Check if we need to separate ksp in this case
PetscErrorCode Simulation::correct_fields()
{
  PetscFunctionBeginUser;
  PetscCall(MatScale(matL, 0.5 * dt));  // matL = dt / 2 * matL
  PetscCall(MatMultAdd(matL, En, currJ, currJ));  // currJ = currJ + matL * E'^{n+1/2}

  PetscCall(VecDot(currJ, En, &w1));  // w1 = (currJ, E'^{n+1/2})
  PetscCall(VecCopy(currJe, currJ));  // currJ = currJe

  // Solving Maxwell's equation to find correct
  // E^{n+1/2}, satisfying continuity equation
  PetscCall(advance_fields(correct, currJe));

  PetscCall(VecDot(currJ, En, &w2));  // w2 = (currJ, E^{n+1/2})

  PetscCall(VecAXPBY(E, -1.0, 2.0, En));  // E^{n+1} = 2 * E^{n+1/2} - E^{n}
  PetscCall(MatMultAdd(rotE, En, B, B));  // B^{n+1} -= dt * rot(E^{n+1/2})
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::advance_fields(KSP ksp, Vec rhs)
{
  PetscFunctionBeginUser;
  PetscCall(VecAXPBY(rhs, 2.0, -dt, E));  // rhs = 2 * E^{n} - (dt * rhs)
  PetscCall(MatMultAdd(rotB, B, rhs, rhs));  // rhs = rhs + rotB(B^{n})

  PetscCall(KSPSolve(ksp, rhs, En));
  PetscCall(KSPGetSolution(ksp, &En));

  // Convergence analysis
  PetscCall(KSPConvergedReasonView(ksp, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(PETSC_SUCCESS);
}


Simulation::~Simulation()
{
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&E));
  PetscCallVoid(VecDestroy(&En));
  PetscCallVoid(VecDestroy(&B));
  PetscCallVoid(VecDestroy(&B0));
  PetscCallVoid(VecDestroy(&currI));
  PetscCallVoid(VecDestroy(&currJ));
  PetscCallVoid(VecDestroy(&currJe));

  // PetscCallVoid(VecDestroy(&charge_density_old));
  // PetscCallVoid(VecDestroy(&charge_density));

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
