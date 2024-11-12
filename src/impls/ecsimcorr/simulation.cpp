#include "simulation.h"

#include "src/utils/operators.h"
#include "src/utils/utils.h"


namespace ecsimcorr {

PetscErrorCode Simulation::initialize_implementation()
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

  /// @todo Should be created on scalar dmda!
  // PetscCall(DMCreateGlobalVector(da, &charge_density_old));
  // PetscCall(DMCreateGlobalVector(da, &charge_density));

  PetscCall(DMCreateMatrix(da, &matL));
  PetscCall(MatDuplicate(matL, MAT_DO_NOT_COPY_VALUES, &matA));

  Divergence divergence(da);
  PetscCall(divergence.create_positive(&divE));

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

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::timestep_implementation(timestep_t timestep)
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
  PetscCall(VecCopy(currI, currJ));

  // The same copying is made for `matL`
  PetscCall(MatCopy(matL, matA, DIFFERENT_NONZERO_PATTERN));  // matA = matL
  PetscCall(MatAYPX(matA, 0.5 * POW2(dt), matM, DIFFERENT_NONZERO_PATTERN));  // matA = matM + dt^2 / 2 * matA

  // Solving the Maxwell's equations to find prediction of E'^{n+1/2}
  PetscCall(advance_fields(currI, matA));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @todo Do we use successive solve here? Check if we need to separate ksp in this case
PetscErrorCode Simulation::correct_fields()
{
  PetscFunctionBeginUser;
  PetscCall(MatScale(matL, 0.5 * dt));  // matL = dt / 2 * matL
  PetscCall(MatMultAdd(matL, En, currJ, currJ));  // currJ = currJ + matL * E'^{n+1/2}

  PetscCall(VecDot(currJ, En, &w1));  // w1 = (currJ, E'^{n+1/2})

  // Solving Maxwell's equation to find correct
  // E^{n+1/2}, satisfying continuity equation
  PetscCall(advance_fields(currJ, matM));

  PetscCall(VecDot(currJe, En, &w2));  // w2 = (currJe, E^{n+1/2})

  PetscCall(VecAXPBY(E, -1.0, 2.0, En));  // E^{n+1} = 2 * E^{n+1/2} - E^{n}
  PetscCall(MatMultAdd(rotE, En, B, B));  // B^{n+1} -= dt * rot(E^{n+1/2})
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::advance_fields(Vec rhs, Mat Amat)
{
  PetscFunctionBeginUser;
  PetscCall(VecAXPBY(rhs, 2.0, -dt, E));  // rhs = 2 * E^{n} - (dt * rhs)
  PetscCall(MatMultAdd(rotB, B, rhs, rhs));  // rhs = rhs + rotB(B^{n})

  PetscCall(KSPSetOperators(ksp, Amat, Amat));
  PetscCall(KSPSolve(ksp, rhs, En));

  /// @todo Convergence analysis, `KSPSolve()` may have diverged
  // KSPConvergedReason reason;
  // PetscCall(KSPGetConvergedReason(ksp, &reason));
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
  PetscCallVoid(MatDestroy(&divE));

  PetscCallVoid(KSPDestroy(&ksp));
  PetscFunctionReturnVoid();
}

}  // namespace ecsimcorr
