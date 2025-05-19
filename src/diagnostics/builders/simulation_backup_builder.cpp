#include "simulation_backup_builder.h"

#include "src/interfaces/simulation.h"
#include "src/utils/configuration.h"
#include "src/utils/vector_utils.h"

SimulationBackupDiagBuilder::SimulationBackupDiagBuilder(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : DiagnosticBuilder(simulation, diagnostics)
{
}

namespace {

auto make_simulation_backup(interfaces::Simulation& simulation, PetscInt period)
{
  return std::make_unique<SimulationBackup>(
    CONFIG().out_dir + "/simulation_backup/",  //
    period,                                    //
    simulation.get_backup_fields(),            //
    simulation.get_backup_particles());
}

}

PetscErrorCode SimulationBackupDiagBuilder::build(
  const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  PetscReal dp;
  info.at("diagnose_period").get_to(dp);

  auto res_dir = CONFIG().out_dir + "/simulation_backup";
  CONFIG().save(res_dir);

  if (info.contains("save_sources") && (bool)info.at("save_sources"))
    CONFIG().save_sources(res_dir + "/src");

  diagnostics_.emplace_back(make_simulation_backup(simulation_, dp));

  LOG("  Simulation backup diagnostic is added, diagnose period: {:.1f} [1/w_pe], {} [dt]", dp * dt, dp);
  PetscFunctionReturn(PETSC_SUCCESS);
}


SimulationBackupCommBuilder::SimulationBackupCommBuilder(
  interfaces::Simulation& simulation, std::vector<Command_up>& commands)
  : CommandBuilder(simulation, commands)
{
}

PetscErrorCode SimulationBackupCommBuilder::build(
  const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  PetscInt load_from;
  info.at("load_from").get_to(load_from);

  LOG("Restoring simulation from backup at {:.1f} [1/w_pe], {} [dt]", load_from * dt, load_from);
  LOG("Other preset commands would be dropped", load_from * dt, load_from);

  /// @note After this all processes will begin from `start`, including first commands call
  simulation_.start = load_from;

  commands_.emplace_back(make_simulation_backup(simulation_, -1));
  PetscFunctionReturn(PETSC_SUCCESS);
}
