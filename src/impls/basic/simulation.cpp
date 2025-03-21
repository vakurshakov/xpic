#include "simulation.h"

#include "src/commands/builders/command_builder.h"
#include "src/diagnostics/builders/diagnostic_builder.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"

namespace basic {

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  DM da = world.da;
  PetscCall(DMCreateGlobalVector(da, &E_));
  PetscCall(DMCreateGlobalVector(da, &B_));
  PetscCall(DMCreateGlobalVector(da, &J_));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rot_dt_p));
  PetscCall(rotor.create_negative(&rot_dt_m));
  PetscCall(MatScale(rot_dt_p, -dt));
  PetscCall(MatScale(rot_dt_m, +dt));

  PetscCall(init_particles());

  std::vector<Command_up> presets;
  PetscCall(build_commands(*this, "Presets", presets));
  PetscCall(build_commands(*this, "StepPresets", step_presets_));

  LOG("Executing presets");
  for (auto&& preset : presets)
    preset->execute(start);

  PetscCall(build_diagnostics(*this, diagnostics_));
  // diagnostics_.emplace_back(std::make_unique<EnergyConservation>(*this));
  // diagnostics_.emplace_back(std::make_unique<ChargeConservation>(*this));

  PetscFunctionReturn(PETSC_SUCCESS);
}


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


PetscErrorCode Simulation::timestep_implementation(PetscInt /* timestep */)
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(J_, 0.0));

  for (auto& sort : particles_) {
    PetscCall(sort->push());
    PetscCall(sort->update_cells());
  }

  Vec util;
  PetscReal norm;
  PetscCall(DMGetGlobalVector(world.da, &util));

  PetscCall(VecSet(util, 0.0));
  PetscCall(VecCopy(E_, util));

  PetscCall(MatMultAdd(rot_dt_p, E_, B_, B_));  // B^{n+1} = B^{n} - rot(E) * dt
  PetscCall(MatMultAdd(rot_dt_m, B_, E_, E_));  // E'^{n+1} = E^{n} + rot(B) * dt
  PetscCall(VecAXPY(E_, -1, J_));  // E^{n+1} = E'^{n+1} - J

  PetscCall(VecAXPY(util, -1, E_));
  PetscCall(VecNorm(util, NORM_2, &norm));
  LOG("  Norm of the difference in electric fields between steps: {:.7f}", norm);

  PetscCall(DMRestoreGlobalVector(world.da, &util));
  PetscFunctionReturn(PETSC_SUCCESS);
}

Vec Simulation::get_named_vector(std::string_view name)
{
  if (name == "E")
    return E_;
  if (name == "B")
    return B_;
  if (name == "J")
    return J_;
  throw std::runtime_error("Unknown vector name " + std::string(name));
}

Particles& Simulation::get_named_particles(std::string_view name)
{
  return interfaces::Simulation::get_named_particles(name, particles_);
}


Simulation::~Simulation()
{
  PetscFunctionBeginUser;
  PetscCallVoid(VecDestroy(&E_));
  PetscCallVoid(VecDestroy(&B_));
  PetscCallVoid(VecDestroy(&J_));
  PetscCallVoid(MatDestroy(&rot_dt_p));
  PetscCallVoid(MatDestroy(&rot_dt_m));
  PetscFunctionReturnVoid();
}

}  // namespace basic
