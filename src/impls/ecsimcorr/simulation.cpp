#include "simulation.h"

#include "src/utils/utils.h"
#include "src/utils/operators.h"


namespace ecsimcorr {

PetscErrorCode Simulation::initialize_implementation() {
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

  PetscCall(DMCreateGlobalVector(da, &charge_density_old));
  PetscCall(DMCreateGlobalVector(da, &charge_density));

  PetscCall(DMCreateMatrix(da, &matL));

  Identity identity(da);
  PetscCall(identity.create(&matI));

  Divergence divergence(da);
  PetscCall(divergence.create_positive(&divE));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rotE));
  PetscCall(rotor.create_negative(&rotB));
  PetscCall(MatScale(rotE, -dt));
  PetscCall(MatScale(rotB, +dt));

  PetscCall(DMCreateMatrix(da, &rot2EB));
  PetscCall(MatMatMult(rotE, rotB, MAT_REUSE_MATRIX, PETSC_CURRENT, &rot2EB));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::timestep_implementation(timestep_t timestep) {
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

Simulation::~Simulation() {
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&E));
  PetscCallVoid(VecDestroy(&En));
  PetscCallVoid(VecDestroy(&Ep));
  PetscCallVoid(VecDestroy(&B));
  PetscCallVoid(VecDestroy(&B0));
  PetscCallVoid(VecDestroy(&currI));
  PetscCallVoid(VecDestroy(&currJ));
  PetscCallVoid(VecDestroy(&currJe));

  PetscCallVoid(VecDestroy(&charge_density_old));
  PetscCallVoid(VecDestroy(&charge_density));

  PetscCallVoid(MatDestroy(&matL));
  PetscCallVoid(MatDestroy(&matI));
  PetscCallVoid(MatDestroy(&rotE));
  PetscCallVoid(MatDestroy(&rotB));
  PetscCallVoid(MatDestroy(&rot2EB));
  PetscCallVoid(MatDestroy(&divE));
  PetscFunctionReturnVoid();
}

}
