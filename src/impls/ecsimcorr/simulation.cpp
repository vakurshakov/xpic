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
  PetscCall(DMCreateGlobalVector(da, &Ep));
  PetscCall(DMCreateGlobalVector(da, &B));
  PetscCall(DMCreateGlobalVector(da, &B0));
  PetscCall(DMCreateGlobalVector(da, &currI));
  PetscCall(DMCreateGlobalVector(da, &currJ));
  PetscCall(DMCreateGlobalVector(da, &currJe));

  /// @todo Should be created on scalar dmda!
  // PetscCall(DMCreateGlobalVector(da, &charge_density_old));
  // PetscCall(DMCreateGlobalVector(da, &charge_density));

  PetscCall(DMCreateMatrix(da, &matL));

  Divergence divergence(da);
  PetscCall(divergence.create_positive(&divE));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rotE));
  PetscCall(rotor.create_negative(&rotB));

  PetscCall(MatProductCreate(rotB, rotE, nullptr, &rot2BE));
  PetscCall(MatProductSetType(rot2BE, MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(rot2BE));
  PetscCall(MatProductSymbolic(rot2BE));
  PetscCall(MatProductNumeric(rot2BE));

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

  // Storing curent before we it as a right hand side of the `ksp`
  PetscCall(VecAXPY(currJ, 1.0, currI));

  PetscCall(predict_E());

  // PetscCall(sort.communicate());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::clear_sources()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(currI, 0.0));
  PetscCall(VecSet(currJ, 0.0));
  PetscCall(VecSet(currJe, 0.0));
  PetscCall(MatZeroEntries(matL));

  for (auto& sort : particles_)
    PetscCall(sort.clear_sources());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::predict_E()
{
  PetscFunctionBeginUser;
  Vec rhs = currI;
  PetscCall(VecAYPX(rhs, -dt, E));  // rhs = E - (dt * currI)
  PetscCall(MatMultAdd(rotB, B, rhs, rhs));  // rhs = rhs + rot(B)

  Mat Amat = matL;
  PetscCall(MatAYPX(Amat, 1.0, rot2BE, DIFFERENT_NONZERO_PATTERN));  // Amat = rot(rot()) + matL
  PetscCall(MatScale(Amat, POW2(0.5 * dt)));  // Amat = dt^2 / 4 * Amat
  PetscCall(MatShift(Amat, 2.0));  // Amat = 2 * I + Amat

  PetscCall(KSPSetOperators(ksp, Amat, Amat));
  PetscCall(KSPSolve(ksp, rhs, En));
  PetscFunctionReturn(PETSC_SUCCESS);
}


Simulation::~Simulation()
{
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&E));
  PetscCallVoid(VecDestroy(&En));
  PetscCallVoid(VecDestroy(&Ep));
  PetscCallVoid(VecDestroy(&B));
  PetscCallVoid(VecDestroy(&B0));
  PetscCallVoid(VecDestroy(&currI));
  PetscCallVoid(VecDestroy(&currJ));
  PetscCallVoid(VecDestroy(&currJe));

  // PetscCallVoid(VecDestroy(&charge_density_old));
  // PetscCallVoid(VecDestroy(&charge_density));

  PetscCallVoid(MatDestroy(&matL));
  PetscCallVoid(MatDestroy(&rotE));
  PetscCallVoid(MatDestroy(&rotB));
  PetscCallVoid(MatDestroy(&rot2BE));
  PetscCallVoid(MatDestroy(&divE));

  PetscCallVoid(KSPDestroy(&ksp));
  PetscFunctionReturnVoid();
}

}  // namespace ecsimcorr
