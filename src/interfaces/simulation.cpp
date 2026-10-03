#include "simulation.h"

#include "src/commands/builders/command_builder.h"
#include "src/diagnostics/builders/diagnostic_builder.h"
#include "src/diagnostics/charge_conservation.h"
#include "src/diagnostics/energy.h"
#include "src/diagnostics/momentum_conservation.h"
#include "src/impls/basic/simulation.h"
#include "src/impls/drift_kinetic/simulation.h"
#include "src/impls/eccapfim/simulation.h"
#include "src/impls/ecsim/simulation.h"
#include "src/impls/ecsimcorr/simulation.h"
#include "src/utils/configuration.h"
#include "src/utils/operators.h"

std::unique_ptr<interfaces::Simulation> build_simulation()
{
  const Configuration::json_t& json = CONFIG().json;

  std::string simulation_str;
  json.at("Simulation").get_to(simulation_str);

  std::unique_ptr<interfaces::Simulation> simulation = nullptr;

  if (simulation_str == "basic")
    simulation = std::make_unique<basic::Simulation>();
  else if (simulation_str == "drift_kinetic")
    simulation = std::make_unique<drift_kinetic::Simulation>();
  else if (simulation_str == "eccapfim")
    simulation = std::make_unique<eccapfim::Simulation>();
  else if (simulation_str == "ecsim")
    simulation = std::make_unique<ecsim::Simulation>();
  else if (simulation_str == "ecsimcorr")
    simulation = std::make_unique<ecsimcorr::Simulation>();
  else
    throw std::runtime_error("Unkown simulation is used: " + simulation_str);

  LOG("Simulation is built, scheme {}", simulation_str);
  return simulation;
}

namespace interfaces {

PetscErrorCode Simulation::initialize()
{
  PetscFunctionBeginUser;
  LOG("Running initialization");

  PetscCall(world.initialize());
  PetscCall(log_information());

  // Registering is idempotent so that several simulations can be created and
  // run within a single process (e.g. comparison tests).
  PetscCall(PetscLogStageGetId("Commands run", &stagenums[0]));
  if (stagenums[0] < 0)
    PetscCall(PetscLogStageRegister("Commands run", &stagenums[0]));

  PetscCall(PetscLogStageGetId("Diagnostics run", &stagenums[1]));
  if (stagenums[1] < 0)
    PetscCall(PetscLogStageRegister("Diagnostics run", &stagenums[1]));

  da = world.da;
  da_rho = world.da_rho;

  PetscCall(DMCreateGlobalVector(da, &E));
  PetscCall(DMCreateGlobalVector(da, &B));
  PetscCall(DMCreateGlobalVector(da, &B0));
  PetscCall(DMCreateGlobalVector(da, &J));
  PetscCall(DMCreateGlobalVector(da, &rE));
  PetscCall(DMCreateGlobalVector(da, &rB));
  PetscCall(DMCreateLocalVector(da, &E_loc));
  PetscCall(DMCreateLocalVector(da, &B_loc));
  PetscCall(DMCreateLocalVector(da, &J_loc));

  PetscCall(PetscObjectSetName((PetscObject)E, "E"));
  PetscCall(PetscObjectSetName((PetscObject)B, "B"));
  PetscCall(PetscObjectSetName((PetscObject)J, "J"));
  PetscCall(PetscObjectSetName((PetscObject)B0, "B0"));

  PetscCall(DMSetMatrixPreallocateOnly(da, PETSC_FALSE));
  PetscCall(DMSetMatrixPreallocateSkip(da, PETSC_TRUE));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rotE));
  PetscCall(rotor.create_negative(&rotB));

  PetscCall(MatProductCreate(rotB, rotE, NULL, &matM));
  PetscCall(MatProductSetType(matM, MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(matM));
  PetscCall(MatProductSymbolic(matM));
  PetscCall(MatProductNumeric(matM));
  PetscCall(MatScale(matM, 0.25 * dt * dt));
  PetscCall(MatShift(matM, 1.0));

  Divergence divergence(da);
  PetscCall(divergence.create_negative(&divE));

  PetscCall(initialize_implementation());

  std::vector<const interfaces::Particles*> particles;
  for (const auto& sort : particles_) {
    particles.emplace_back(sort.get());
  }

  /// @todo Use a shared pointers for diagnostics
  /// @todo Or else store the `TableDiagnostics` themselves
  bool append_energy_conservation = true;

  for (auto& diagnostic : diagnostics_) {
    if (dynamic_cast<Energy*>(diagnostic.get()))
      append_energy_conservation = false;
  }

  if (append_energy_conservation) {
    diagnostics_.emplace_back( //
      std::make_unique<Energy>(*this));
  }

  diagnostics_.emplace_back( //
    std::make_unique<ChargeConservation>(*this));

  diagnostics_.emplace_back( //
    std::make_unique<MomentumConservation>(da, E, particles));

  std::vector<Command_up> presets;
  PetscCall(build_commands(*this, "Presets", presets));
  PetscCall(build_commands(*this, "StepPresets", step_presets_));
  PetscCall(build_diagnostics(*this, diagnostics_));

  LOG("Executing presets");
  for (const Command_up& preset : presets)
    preset->execute(start);

  for (const Particles_sp& sort : particles_)
    PetscCall(sort->initialize_rho());

  for (const Diagnostic_up& diagnostic : diagnostics_)
    PetscCall(diagnostic->diagnose(start));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::calculate()
{
  PetscFunctionBeginUser;
  LOG("Running the main simulation cycle");
  for (PetscInt t = start + 1; t <= geom_nt; ++t) {
    LOG_FLUSH();
    LOG("Timestep = {:.4f} [1/w_pe] = {} [dt]", t * dt, t);

    PetscCall(PetscLogStagePush(stagenums[0]));
    for (const Command_up& command : step_presets_)
      PetscCall(command->execute(t));
    PetscCall(PetscLogStagePop());

    PetscCall(timestep_implementation(t));

    for (const Particles_sp& sort : particles_)
      PetscCall(sort->calculate_rho());

    PetscCall(PetscLogStagePush(stagenums[1]));
    for (const Diagnostic_up& diagnostic : diagnostics_)
      PetscCall(diagnostic->diagnose(t));
    PetscCall(PetscLogStagePop());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::finalize()
{
  PetscFunctionBeginUser;
  for (const Command_up& command : step_presets_)
    PetscCall(command->finalize());

  for (const Diagnostic_up& diagnostic : diagnostics_)
    PetscCall(diagnostic->finalize());

  for (const Particles_sp& sort : particles_)
    PetscCall(sort->finalize());

  PetscCall(VecDestroy(&E));
  PetscCall(VecDestroy(&B));
  PetscCall(VecDestroy(&B0));
  PetscCall(VecDestroy(&J));
  PetscCall(VecDestroy(&E_loc));
  PetscCall(VecDestroy(&B_loc));
  PetscCall(VecDestroy(&J_loc));
  PetscCall(VecDestroy(&rE));
  PetscCall(VecDestroy(&rB));

  PetscCall(MatDestroy(&rotE));
  PetscCall(MatDestroy(&rotB));
  PetscCall(MatDestroy(&matM));
  PetscCall(MatDestroy(&divE));

  PetscCall(world.finalize());
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


Particles& Simulation::get_named_particles(std::string_view name)
{
  auto it = std::find_if(particles_.begin(), particles_.end(), //
    [&](const Particles_sp& sort) {
      return sort->parameters.sort_name == name;
    });

  if (it == particles_.end())
    throw std::runtime_error("No particles with name " + std::string(name));
  return **it;
}

}  // namespace interfaces
