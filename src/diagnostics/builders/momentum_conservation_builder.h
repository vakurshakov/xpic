#ifndef SRC_DIAGNOSTICS_BUILDERS_MOMENTUM_CONSERVATION_BUILDER_H
#define SRC_DIAGNOSTICS_BUILDERS_MOMENTUM_CONSERVATION_BUILDER_H

#include "src/diagnostics/builders/diagnostic_builder.h"

class MomentumConservationBuilder : public DiagnosticBuilder {
public:
  MomentumConservationBuilder(interfaces::Simulation& simulation,
    std::vector<Diagnostic_up>& diagnostics);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the momentum_conservation diagnostic description:\n"
      "{\n"
      "  \"diagnostic\": \"momentum_conservation\", -- Name of the diagnostic, constant.\n"
      "}\n"
      "The balance is exact only for the \"basic\" implementation.";
    return help;
  }
};

#endif  // SRC_DIAGNOSTICS_BUILDERS_MOMENTUM_CONSERVATION_BUILDER_H
