#include "simulation.h"

#include "src/commands/builders/command_builder.h"
#include "src/diagnostics/builders/diagnostic_builder.h"
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
  PetscCall(DMCreateGlobalVector(da, &J));
  PetscCall(DMCreateLocalVector(da, &local_E));
  PetscCall(DMCreateLocalVector(da, &local_B));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rot_dt_p));
  PetscCall(rotor.create_negative(&rot_dt_m));

  // For the reasoning see `timestep_implementation()`
  PetscCall(MatScale(rot_dt_p, -(0.5 * dt)));
  PetscCall(MatScale(rot_dt_m, +(1.0 * dt)));

  PetscCall(init_particles());

  std::vector<Command_up> presets;
  PetscCall(build_commands(*this, "Presets", presets));
  PetscCall(build_commands(*this, "StepPresets", step_presets_));

  LOG("Executing presets");
  for (auto&& preset : presets)
    preset->execute(start);

  PetscCall(build_diagnostics(*this, diagnostics_));

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

  diagnostics_.emplace_back(
    std::make_unique<ChargeConservation>(world.da, currents, sorts));
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
  // B^{n} = B^{n-1/2} - rot(E^{n}) * (0.5 * dt)
  PetscCall(MatMultAdd(rot_dt_p, E, B, B));

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

  // B^{n+1/2} = B^{n} - rot(E^{n}) * (0.5 * dt)
  PetscCall(MatMultAdd(rot_dt_p, E, B, B));

  // E'^{n+1} = E^{n} + rot(B^{n+1/2}) * dt
  PetscCall(MatMultAdd(rot_dt_m, B, E, E));

  // E^{n+1} = E'^{n+1} - J
  PetscCall(VecAXPY(E, -1, J));

  PetscCall(VecAXPY(util, -1, E));
  PetscCall(VecNorm(util, NORM_2, &norm));
  LOG("  Norm of the difference in electric fields between steps: {:.7f}", norm);

  PetscCall(DMRestoreGlobalVector(world.da, &util));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @todo Is it a `Command`? Where it should be placed?
PetscErrorCode Simulation::init_particles()
{
  PetscFunctionBeginUser;
  LOG("Configuring particles");

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

    PetscReal T = std::hypot(parameters.Tx, parameters.Ty, parameters.Tz);
    PetscReal V = std::sqrt(T / (parameters.m * 511.0));
    PetscReal H = std::hypot(Dx[X] / V, Dx[Y] / V, Dx[Z] / V);

    LOG("  {} are added:", parameters.sort_name);
    LOG("    temperature,         T = {:.3e} [KeV]", T);
    LOG("    thermal velocity, v_th = {:.3e} [c]", V);
    LOG("    cell-heating, Dx / L_d = {:.3e} [unit]", H);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

Vec Simulation::get_named_vector(std::string_view name)
{
  if (name == "E")
    return E;
  if (name == "B")
    return B;
  if (name == "J")
    return J;
  throw std::runtime_error("Unknown vector name " + std::string(name));
}

Particles& Simulation::get_named_particles(std::string_view name)
{
  return interfaces::Simulation::get_named_particles(name, particles_);
}


Simulation::~Simulation()
{
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&E));
  PetscCallVoid(VecDestroy(&B));
  PetscCallVoid(VecDestroy(&J));
  PetscCallVoid(VecDestroy(&local_E));
  PetscCallVoid(VecDestroy(&local_B));
  PetscCallVoid(MatDestroy(&rot_dt_p));
  PetscCallVoid(MatDestroy(&rot_dt_m));
  PetscFunctionReturnVoid();
}

}  // namespace basic
