#include "simulation.h"

#include "src/impls/basic/simulation.h"
#include "src/impls/ecsimcorr/simulation.h"
#include "src/impls/ricketson/simulation.h"
#include "src/utils/configuration.h"

namespace interfaces {

PetscErrorCode Simulation::initialize()
{
  PetscFunctionBeginUser;
  PetscCall(world.initialize());
  PetscCall(log_information());
  PetscCall(initialize_implementation());

  for (const Diagnostic_up& diagnostic : diagnostics_)
    PetscCall(diagnostic->diagnose(0));

  PetscLogStageRegister("Commands run", &stagenums[0]);
  PetscLogStageRegister("Diagnostics run", &stagenums[1]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::log_information() const
{
  PetscFunctionBeginUser;
  static const PetscReal n0 = sqrt(1e13);
  LOG("Note: Dimensionless units are used.");
  LOG("For reference, using density 1e13 cm^(-3):");
  LOG("  frequency,   w_pe = {} [1/sec]", 5.64e+4 * n0);
  LOG("  time,      1/w_pe = {} [sec]", 1.77e-5 / n0);
  LOG("  length,    c/w_pe = {} [cm]", 5.32e+5 / n0);
  LOG("  electric field, E = {} [MV/cm]", 9.63e-7 * n0);
  LOG("  magnetic field, B = {} [T]", 3.21e-7 * n0);

  LOG("Geometric constants for the current setup:");
  LOG("  (length along x axis) = {} [c/w_pe] = {} [dx]", geom_x, geom_nx);
  LOG("  (length along y axis) = {} [c/w_pe] = {} [dy]", geom_y, geom_ny);
  LOG("  (length along z axis) = {} [c/w_pe] = {} [dz]", geom_z, geom_nz);
  LOG("  (simulation time)     = {} [1/w_pe] = {} [dt]", geom_t, geom_nt);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::calculate()
{
  PetscFunctionBeginUser;
  for (PetscInt t = start_ + 1; t <= geom_nt; ++t) {
    LOG_FLUSH();
    LOG("Timestep = {:.4f} [1/w_pe] = {} [dt]", t * dt, t);

    PetscLogStagePush(stagenums[0]);
    for (const Command_up& command : step_presets_)
      PetscCall(command->execute(t));
    PetscLogStagePop();

    PetscCall(timestep_implementation(t));

    PetscLogStagePush(stagenums[1]);
    for (const Diagnostic_up& diagnostic : diagnostics_)
      PetscCall(diagnostic->diagnose(t));
    PetscLogStagePop();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace interfaces


Simulation_up build_simulation()
{
  Simulation_up simulation = nullptr;

  std::string simulation_str;

  const Configuration& config = CONFIG();
  config.json.at("Simulation").get_to(simulation_str);

  if (simulation_str == "basic")
    return std::make_unique<basic::Simulation>();
  if (simulation_str == "ricketson")
    return std::make_unique<ricketson::Simulation>();
  if (simulation_str == "ecsimcorr")
    return std::make_unique<ecsimcorr::Simulation>();

  throw std::runtime_error("Unkown simulation is used: " + simulation_str);
}
