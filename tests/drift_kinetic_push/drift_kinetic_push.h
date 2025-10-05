#include "src/algorithms/drift_kinetic_push.h"
#include "src/algorithms/drift_kinetic_implicit.h" 
#include "src/utils/vector3.h"
#include "tests/common.h"
#include "src/utils/world.h"
#include "src/utils/utils.h"

PetscErrorCode get_omega_dt(PetscReal& omega_dt)
{
  PetscFunctionBeginUser;
  PetscBool flg;
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-omega_dt", &omega_dt, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must specify the timestep (Omega * dt) with '-omega_dt'");
  PetscFunctionReturn(PETSC_SUCCESS);
}
