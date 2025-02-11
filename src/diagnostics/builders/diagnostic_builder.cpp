#include "diagnostic_builder.h"

#include "src/diagnostics/builders/distribution_moment_builder.h"
#include "src/diagnostics/builders/field_view_builder.h"
#include "src/utils/geometries.h"

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

  using namespace interfaces;

  for (auto&& info : diagnostics) {
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
    else {
      throw std::runtime_error("Unknown diagnostic name " + name);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

