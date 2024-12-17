#ifndef SRC_BASIC_BUILDERS_FIELDS_ENERGY_BUILDER_H
#define SRC_BASIC_BUILDERS_FIELDS_ENERGY_BUILDER_H

#include "src/diagnostics/fields_energy.h"
#include "src/impls/basic/builders/diagnostic_builder.h"

class FieldsEnergyBuilder : public DiagnosticBuilder {
public:
  FieldsEnergyBuilder(const interfaces::Simulation& simulation,
    std::vector<Diagnostic_up>& diagnostics)
    : DiagnosticBuilder(simulation, diagnostics)
  {
  }

  PetscErrorCode build(const Configuration::json_t& /* diag_info */) override
  {
    PetscFunctionBeginUser;
    diagnostics_.emplace_back(
      std::make_unique<FieldsEnergy>(CONFIG().out_dir + "/",
        simulation_.world_.da, get_global_vector("E"), get_global_vector("B")));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  std::string_view usage_message() const override
  {
    return "\n"
           "Usage: Simply add `\"fields_energy\": {}` into configuration file, "
           "no addition description is needed.";
  }
};

#endif  // SRC_BASIC_BUILDERS_FIELDS_ENERGY_BUILDER_H
