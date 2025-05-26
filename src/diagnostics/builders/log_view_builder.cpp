#include "log_view_builder.h"

#include "src/diagnostics/log_view.h"
#include "src/utils/configuration.h"

LogViewBuilder::LogViewBuilder(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : DiagnosticBuilder(simulation, diagnostics)
{
}

PetscErrorCode LogViewBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  std::string level_info;
  info.at("level").get_to(level_info);

  LogView::Level level = LogView::EachTimestep;

  if (level_info == "EachTimestep")
    level = LogView::EachTimestep;
  else if (level_info == "DiagnosePeriodAvg")
    level = LogView::DiagnosePeriodAvg;
  else if (level_info == "AllTimestepsSummary")
    level = LogView::AllTimestepsSummary;

  diagnostics_.emplace_back(std::make_unique<LogView>(level));

  LOG("  LogView with {} diagnostic is added", level_info);
  PetscFunctionReturn(PETSC_SUCCESS);
}
