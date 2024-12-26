#include "diagnostic_builder.h"

#include "src/diagnostics/builders/distribution_moment_builder.h"
#include "src/diagnostics/builders/field_view_builder.h"
#include "src/diagnostics/builders/fields_energy_builder.h"
#include "src/utils/region_operations.h"

namespace {

template<class InheritedBuilderType>
PetscErrorCode build_diagnostic(const Configuration::json_t& info,
  const interfaces::Simulation& simulation, std::vector<Diagnostic_up>& result)
{
  PetscFunctionBeginUser;
  auto&& builder = std::make_unique<InheritedBuilderType>(simulation, result);
  PetscCall(builder->build(info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace


DiagnosticBuilder::DiagnosticBuilder(const interfaces::Simulation& simulation,
  std::vector<Diagnostic_up>& diagnostics)
  : Builder(simulation), diagnostics_(diagnostics)
{
}


PetscErrorCode build_diagnostics(
  const interfaces::Simulation& simulation, std::vector<Diagnostic_up>& result)
{
  const Configuration::json_t& diagnostics = CONFIG().json.at("Diagnostics");

  if (diagnostics.empty())
    return PETSC_SUCCESS;

  PetscFunctionBeginUser;
  LOG("Building diagnostics");

  for (auto&& info : diagnostics) {
    std::string name;
    info.at("diagnostic").get_to(name);

    if (name == "FieldsEnergy") {
      // PetscCall(build_diagnostic<FieldsEnergyBuilder>(info, simulation, result));
      continue;
    }
    else if (name == "FieldView") {
      PetscCall(build_diagnostic<FieldViewBuilder>(info, simulation, result));
    }
    else if (name == "DistributionMoment") {
      // PetscCall(build_diagnostic<DistributionMomentBuilder>(info, simulation, result));
      continue;
    }
    else {
      throw std::runtime_error("Unknown diagnostic name: " + name);
    }
  }

  /// @todo Check uniqueness of result directories
  PetscFunctionReturn(PETSC_SUCCESS);
}

