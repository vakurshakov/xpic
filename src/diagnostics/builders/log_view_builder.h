#ifndef SRC_BASIC_BUILDERS_LOG_VIEW_BUILDER_H
#define SRC_BASIC_BUILDERS_LOG_VIEW_BUILDER_H

#include "src/diagnostics/builders/diagnostic_builder.h"

class LogViewBuilder : public DiagnosticBuilder {
public:
  LogViewBuilder(const interfaces::Simulation& simulation,
    std::vector<Diagnostic_up>& diagnostics);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the LogView diagnostic description:\n"
      "{\n"
      "  \"diagnostic\": \"LogView\", -- Name of the diagnostic, constant."
      "  \"level\": \"LevelName\", -- Logging level, one of the following:\n"
      "               EachTimestep, DiagnosePeriodAvg, AllTimestepsSummary\n"
      "}";
    return help;
  }
};

#endif  // SRC_BASIC_BUILDERS_LOG_VIEW_BUILDER_H
