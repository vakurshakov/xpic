#include "simulation.h"

#include "src/diagnostics/charge_conservation.h"
#include "src/diagnostics/energy_conservation.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"

namespace basic {

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  DM da = world.da;
  PetscCall(DMCreateGlobalVector(da, &E));
  PetscCall(DMCreateGlobalVector(da, &B));
  PetscCall(DMCreateGlobalVector(da, &B0));
  PetscCall(DMCreateGlobalVector(da, &J));
  PetscCall(DMCreateLocalVector(da, &local_E));
  PetscCall(DMCreateLocalVector(da, &local_B));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rot_dt_p));
  PetscCall(rotor.create_negative(&rot_dt_m));

  // For the reasoning see `timestep_implementation()`
  PetscCall(MatScale(rot_dt_p, -(0.5 * dt)));
  PetscCall(MatScale(rot_dt_m, +(1.0 * dt)));

  PetscCall(init_particles(*this, particles_));

  if (!CONFIG().is_loaded_from_backup())
    PetscCall(VecAXPY(B, 1.0, B0));

  std::vector<Vec> currents;
  std::vector<const interfaces::Particles*> sorts;
  for (const auto& sort : particles_) {
    currents.emplace_back(sort->global_J);
    sorts.emplace_back(sort.get());
  }
  currents.emplace_back(J);

  auto&& f_diag = std::make_unique<FieldsEnergy>(world.da, E, B);
  auto&& p_diag = std::make_unique<ParticlesEnergy>(sorts);

  diagnostics_.emplace_back(std::make_unique<EnergyConservation>(
    *this, std::move(f_diag), std::move(p_diag)));

  if (!currents.empty() && !sorts.empty()) {
    diagnostics_.emplace_back(
      std::make_unique<ChargeConservation>(world.da, currents, sorts));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::timestep_implementation(PetscInt /* timestep */)
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(J, 0.0));

  for (auto& sort : particles_) {
    PetscCall(VecSet(sort->global_J, 0.0));
    PetscCall(VecSet(sort->local_J, 0.0));
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
  PetscCall(MatMultAdd(rot_dt_p, E, B, B));
  PetscCall(VecAXPY(B, +1.0, B0));

  if (particles_.empty())
    return PETSC_SUCCESS;

  DM da = world.da;
  PetscCall(DMGlobalToLocal(da, E, INSERT_VALUES, local_E));
  PetscCall(DMGlobalToLocal(da, B, INSERT_VALUES, local_B));

  Vector3R*** arr_E;
  Vector3R*** arr_B;
  PetscCall(DMDAVecGetArrayRead(da, local_E, &arr_E));
  PetscCall(DMDAVecGetArrayRead(da, local_B, &arr_B));

  for (auto& sort : particles_) {
    sort->E = arr_E;
    sort->B = arr_B;
    PetscCall(sort->push());
    PetscCall(sort->update_cells());
  }

  PetscCall(DMDAVecRestoreArrayRead(da, local_E, &arr_E));
  PetscCall(DMDAVecRestoreArrayRead(da, local_B, &arr_B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::push_fields()
{
  PetscFunctionBeginUser;
  Vec util;
  PetscReal norm;
  PetscCall(DMGetGlobalVector(world.da, &util));

  PetscCall(VecSet(util, 0.0));
  PetscCall(VecCopy(E, util));
  PetscCall(VecAXPY(B, -1.0, B0));

  // B^{n+1/2} = B^{n} - rot(E^{n}) * (0.5 * dt)
  PetscCall(MatMultAdd(rot_dt_p, E, B, B));

  // E'^{n+1} = E^{n} + rot(B^{n+1/2}) * dt
  PetscCall(MatMultAdd(rot_dt_m, B, E, E));
  PetscCall(VecAXPY(B, +1.0, B0));

  // E^{n+1} = E'^{n+1} - J
  PetscCall(VecAXPY(E, -1, J));

  PetscCall(VecAXPY(util, -1, E));
  PetscCall(VecNorm(util, NORM_2, &norm));
  LOG("  Norm of the difference in electric fields between steps: {:.7f}", norm);

  PetscCall(DMRestoreGlobalVector(world.da, &util));
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
  PetscCall(VecDestroy(&local_E));
  PetscCall(VecDestroy(&local_B));
  PetscCall(MatDestroy(&rot_dt_p));
  PetscCall(MatDestroy(&rot_dt_m));
  PetscFunctionReturn(PETSC_SUCCESS);
}

Vec Simulation::get_named_vector(std::string_view name) const
{
  static const std::unordered_map<std::string_view, Vec> map{
    {"E", E},
    {"B", B},
    {"B0", B0},
    {"J", J},
  };
  return map.at(name);
}

Simulation::NamedValues<Vec> Simulation::get_backup_fields() const
{
  return {{"E", E}, {"B", B}, {"B0", B0}};
}

}  // namespace basic
