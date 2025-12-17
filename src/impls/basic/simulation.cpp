#include "simulation.h"

#include "src/diagnostics/charge_conservation.h"
#include "src/diagnostics/energy_conservation.h"
#include "src/diagnostics/momentum_conservation.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"

namespace basic {

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(da, &E));
  PetscCall(DMCreateGlobalVector(da, &B));
  PetscCall(DMCreateGlobalVector(da, &B0));
  PetscCall(DMCreateGlobalVector(da, &J));
  PetscCall(DMCreateLocalVector(da, &E_loc));
  PetscCall(DMCreateLocalVector(da, &B_loc));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rotE));
  PetscCall(rotor.create_negative(&rotB));

  // For the reasoning see `timestep_implementation()`
  PetscCall(MatScale(rotE, -(0.5 * dt)));
  PetscCall(MatScale(rotB, +(1.0 * dt)));

  PetscCall(init_particles(*this, particles_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::timestep_implementation(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(J, 0.0));

  for (auto& sort : particles_) {
    PetscCall(VecSet(sort->J, 0.0));
    PetscCall(VecSet(sort->J_loc, 0.0));
  }

  PetscCall(push_particles());
  PetscCall(push_fields());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::push_particles()
{
  PetscFunctionBeginUser;
  PetscCall(VecAXPY(B, -1.0, B0));
  // B^{n} = B^{n-1/2} - rot(E^{n}) * (0.5 * dt)
  PetscCall(MatMultAdd(rotE, E, B, B));
  PetscCall(VecAXPY(B, +1.0, B0));

  if (particles_.empty())
    return PETSC_SUCCESS;

  PetscCall(DMGlobalToLocal(da, E, INSERT_VALUES, E_loc));
  PetscCall(DMGlobalToLocal(da, B, INSERT_VALUES, B_loc));

  PetscCall(DMDAVecGetArrayRead(da, E_loc, &E_arr));
  PetscCall(DMDAVecGetArrayRead(da, B_loc, &B_arr));

  for (auto& sort : particles_) {
    sort->E_arr = E_arr;
    sort->B_arr = B_arr;
    PetscCall(sort->push());
    PetscCall(sort->update_cells());
  }

  PetscCall(DMDAVecRestoreArrayRead(da, E_loc, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, B_loc, &B_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::push_fields()
{
  PetscFunctionBeginUser;
  Vec util;
  PetscReal norm;
  PetscCall(DMGetGlobalVector(da, &util));

  PetscCall(VecSet(util, 0.0));
  PetscCall(VecCopy(E, util));
  PetscCall(VecAXPY(B, -1.0, B0));

  // B^{n+1/2} = B^{n} - rot(E^{n}) * (0.5 * dt)
  PetscCall(MatMultAdd(rotE, E, B, B));

  // E^{n+1} = E^{n} + rot(B^{n+1/2}) * dt - J * dt
  PetscCall(MatMultAdd(rotB, B, E, E));
  PetscCall(VecAXPY(E, -dt, J));

  PetscCall(VecAXPY(B, +1.0, B0));

  PetscCall(VecAXPY(util, -1, E));
  PetscCall(VecNorm(util, NORM_2, &norm));
  LOG("  Norm of the difference in electric fields between steps: {:.7f}", norm);

  PetscCall(DMRestoreGlobalVector(da, &util));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(interfaces::Simulation::finalize());
  PetscCall(VecDestroy(&E));
  PetscCall(VecDestroy(&B));
  PetscCall(VecDestroy(&B0));
  PetscCall(VecDestroy(&J));
  PetscCall(VecDestroy(&E_loc));
  PetscCall(VecDestroy(&B_loc));
  PetscCall(MatDestroy(&rotE));
  PetscCall(MatDestroy(&rotB));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace basic
