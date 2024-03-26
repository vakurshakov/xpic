#ifndef SRC_BASIC_DIAGNOSTICS_BUILDERS_FIELDS_ENERGY_BUILDER_H
#define SRC_BASIC_DIAGNOSTICS_BUILDERS_FIELDS_ENERGY_BUILDER_H

#include "src/impls/basic/diagnostics/builders/diagnostic_builder.h"
#include "src/impls/basic/diagnostics/fields_energy.h"

namespace basic {

class Fields_energy_builder : public Diagnostic_builder {
public:
  Fields_energy_builder(const Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
    : Diagnostic_builder(simulation, diagnostics) {}

  PetscErrorCode build(const Configuration::json_t& /* diag_info */) override {
    PetscFunctionBeginUser;
    diagnostics_.emplace_back(std::make_unique<Fields_energy>(CONFIG().out_dir + "/", simulation_.da_, simulation_.E_, simulation_.B_));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

private:
  constexpr std::string usage_message() const override {
    return "\n" "Usage: Simply add `\"fields_energy\": {}` into configuration file, no addition description is needed.";
  }
};

}

#endif  // SRC_BASIC_DIAGNOSTICS_BUILDERS_FIELDS_ENERGY_BUILDER_H
