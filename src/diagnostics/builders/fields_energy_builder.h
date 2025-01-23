#ifndef SRC_DIAGNOSTICS_BUILDERS_FIELDS_ENERGY_BUILDER_H
#define SRC_DIAGNOSTICS_BUILDERS_FIELDS_ENERGY_BUILDER_H

#include "src/diagnostics/builders/diagnostic_builder.h"

class FieldsEnergyBuilder : public DiagnosticBuilder {
public:
  FieldsEnergyBuilder(const interfaces::Simulation& simulation,
    std::vector<Diagnostic_up>& diagnostics)
    : DiagnosticBuilder(simulation, diagnostics)
  {
  }

  PetscErrorCode build(const Configuration::json_t& /* info */) override
  {
    return PETSC_SUCCESS;
  }

  std::string_view usage_message() const override
  {
    return "";
  }
};

#endif  // SRC_DIAGNOSTICS_BUILDERS_FIELDS_ENERGY_BUILDER_H
