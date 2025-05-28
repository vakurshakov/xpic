#include "diagnostic_builder.h"

#include "src/diagnostics/builders/distribution_moment_builder.h"
#include "src/diagnostics/builders/field_view_builder.h"
#include "src/diagnostics/builders/log_view_builder.h"
#include "src/diagnostics/builders/simulation_backup_builder.h"
#include "src/diagnostics/builders/velocity_distribution_builder.h"
#include "src/utils/geometries.h"

DiagnosticBuilder::DiagnosticBuilder(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : Builder(simulation), diagnostics_(diagnostics)
{
}


PetscErrorCode build_diagnostics(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& result)
{
  PetscFunctionBeginUser;
  using namespace interfaces;

  const Configuration::json_t& config = CONFIG().json;

  LOG("Building diagnostics");

  auto&& it = config.find("SimulationBackup");
  if (it != config.end() && !it->empty())
    PetscCall(Builder::use_impl<SimulationBackupDiagBuilder>(*it, simulation, result));

  it = config.find("Diagnostics");
  if (it == config.end() || it->empty())
    PetscFunctionReturn(PETSC_SUCCESS);

  for (auto&& info : *it) {
    if (!info.contains("diagnostic"))
      continue;

    std::string name;
    info.at("diagnostic").get_to(name);

    if (name == "FieldsEnergy") {
      // PetscCall(build_diagnostic<FieldsEnergyBuilder>(info, simulation, result));
      continue;
    }
    else if (name == "FieldView") {
      PetscCall(Builder::use_impl<FieldViewBuilder>(info, simulation, result));
    }
    else if (name == "DistributionMoment") {
      PetscCall(Builder::use_impl<DistributionMomentBuilder>(info, simulation, result));
    }
    else if (name == "VelocityDistribution") {
      PetscCall(Builder::use_impl<VelocityDistributionBuilder>(info, simulation, result));
    }
    else if (name == "LogView") {
      PetscCall(Builder::use_impl<LogViewBuilder>(info, simulation, result));
    }
    else {
      throw std::runtime_error("Unknown diagnostic name " + name);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

