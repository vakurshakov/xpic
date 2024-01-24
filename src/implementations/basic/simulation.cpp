#include "simulation.h"

#include <petscdmda.h>

namespace basic {

PetscErrorCode Simulation::initialize_implementation() {
  PetscFunctionBeginUser;
  const Configuration& config = CONFIG();

  constexpr PetscInt dof = 3;  // for vector fields
  constexpr PetscInt s = 1;    // stencil width (should depend on particle size)

  // We can specify in our config DMBoundaryType and procs number and map it to Create3d
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, R3C(DM_BOUNDARY_NONE), DMDA_STENCIL_BOX,
    R3CX(config.size_n), R3C(PETSC_DECIDE), dof, s, R3C(nullptr), &da_));
  PetscCall(DMSetUp(da_));

  PetscCall(DMCreateGlobalVector(da_, &E_));
  PetscCall(DMCreateGlobalVector(da_, &B_));
  PetscCall(DMCreateMatrix(da_, &gradient_));
  PetscCall(DMCreateMatrix(da_, &rotor_));

  PetscCall(setup_gradient());
  PetscCall(setup_rotor());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::setup_gradient() {
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::setup_rotor() {
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::timestep_implementation(timestep_t timestep) {
  PetscFunctionReturn(PETSC_SUCCESS);
}

Simulation::~Simulation() {
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&E_));
  PetscCallVoid(VecDestroy(&B_));
  PetscCallVoid(MatDestroy(&gradient_));
  PetscCallVoid(MatDestroy(&rotor_));
  PetscFunctionReturnVoid();
}

}
