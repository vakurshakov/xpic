#ifndef SRC_DIAGNOSTICS_BUILDERS_SIMULATION_BACKUP_BUILDER_H
#define SRC_DIAGNOSTICS_BUILDERS_SIMULATION_BACKUP_BUILDER_H

#include "src/commands/builders/command_builder.h"
#include "src/diagnostics/builders/diagnostic_builder.h"
#include "src/diagnostics/simulation_backup.h"

namespace {

std::string_view help =
  "\nStructure of the SimulationBackup description:\n"
  "{\n"
  "  \"diagnose_period\": X, -- Period between consecutive backups in\n"
  "                          _1/w_pe_ units, this parameter is required.\n"
  "  \"load_from\": Y -- Timestep in _dt_ units to load backups from.\n"
  "                      Optional as it is needed only to force load.\n"
  "}";

}

class SimulationBackupDiagBuilder : public DiagnosticBuilder {
public:
  SimulationBackupDiagBuilder(interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    return help;
  }
};


class SimulationBackupCommBuilder : public CommandBuilder {
public:
  SimulationBackupCommBuilder(
    interfaces::Simulation& simulation, std::vector<Command_up>& commands);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    return help;
  }
};

#endif  // SRC_DIAGNOSTICS_BUILDERS_SIMULATION_BACKUP_BUILDER_H
