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

  LOG("Running initialize implementation");
  PetscCall(initialize_implementation());

  PetscLogStageRegister("Commands run", &stagenums[0]);
  PetscLogStageRegister("Diagnostics run", &stagenums[1]);

  for (const Diagnostic_up& diagnostic : diagnostics_)
    PetscCall(diagnostic->diagnose(start));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::calculate()
{
  PetscFunctionBeginUser;
  for (PetscInt t = start + 1; t <= geom_nt; ++t) {
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

PetscErrorCode Simulation::log_information() const
{
  PetscFunctionBeginUser;
  const PetscReal n0 = std::sqrt(1e13);
  LOG("Note: Dimensionless units are used.");
  LOG("For reference, using density 1e13 cm^(-3):");
  LOG("  frequency,   w_pe = {:4e} [1/sec]", 5.64e+4 * n0);
  LOG("  time,      1/w_pe = {:4e} [sec]",   1.77e-5 / n0);
  LOG("  length,    c/w_pe = {:4e} [cm]",    5.32e+5 / n0);
  LOG("  electric field, E = {:4e} [MV/cm]", 9.63e-7 * n0);
  LOG("  magnetic field, B = {:4e} [T]",     3.21e-7 * n0);

  LOG("Geometric constants for the current setup:");
  LOG("  Lx = {:.2f} [c/w_pe], Nx = {} [dx], dx = {:.2f}", geom_x, geom_nx, dx);
  LOG("  Ly = {:.2f} [c/w_pe], Ny = {} [dy], dy = {:.2f}", geom_y, geom_ny, dy);
  LOG("  Lz = {:.2f} [c/w_pe], Nz = {} [dz], dz = {:.2f}", geom_z, geom_nz, dz);
  LOG("  Lt = {:.2f} [1/w_pe], Nt = {} [dt], dt = {:.2f}", geom_t, geom_nt, dt);
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
