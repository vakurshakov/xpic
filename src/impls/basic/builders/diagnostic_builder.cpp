#include "diagnostic_builder.h"

#include "src/impls/basic/builders/distribution_moment_builder.h"
#include "src/impls/basic/builders/field_view_builder.h"
#include "src/impls/basic/builders/fields_energy_builder.h"
#include "src/utils/region_operations.h"


DiagnosticBuilder::DiagnosticBuilder(const interfaces::Simulation& simulation,
  std::vector<Diagnostic_up>& diagnostics)
  : Builder(simulation), diagnostics_(diagnostics)
{
}


const basic::Particles& DiagnosticBuilder::get_sort(const std::string& name) const
{
  /// @todo This dynamic casting can be removed by direct Simulation class call.
  auto simulation = dynamic_cast<const basic::Simulation*>(&simulation_);
  const std::vector<basic::Particles>& particles = simulation->particles_;

  auto it = std::find_if(particles.begin(), particles.end(),  //
    [&](const basic::Particles& sort) {
      return sort.parameters().sort_name == name;
    });

  if (it == particles.end())
    throw std::runtime_error("No particles with name " + name);
  return *it;
}

PetscErrorCode build_diagnostics(
  const interfaces::Simulation& simulation, std::vector<Diagnostic_up>& result)
{
  PetscFunctionBeginUser;

  LOG("Building diagnostics");
  const Configuration& config = CONFIG();
  const Configuration::json_t& descriptions = config.json.at("Diagnostics");

  Builder_up builder = nullptr;

  for (const auto& [diag_name, diag_info] : descriptions.items()) {
    if (diag_name == "fields_energy") {
      builder = std::make_unique<FieldsEnergyBuilder>(simulation, result);
      PetscCall(builder->build(diag_info));
    }
    else if (diag_name == "field_view") {
      builder = std::make_unique<FieldViewBuilder>(simulation, result);
      PetscCall(builder->build(diag_info));
    }
    else if (  //
      diag_name == "density" || diag_name == "Vx_moment" ||
      diag_name == "Vy_moment" || diag_name == "Vz_moment" ||
      diag_name == "Vr_moment" || diag_name == "Vphi_moment" ||
      diag_name == "mVxVx_moment" || diag_name == "mVxVy_moment" ||
      diag_name == "mVxVz_moment" || diag_name == "mVyVy_moment" ||
      diag_name == "mVyVz_moment" || diag_name == "mVzVz_moment" ||
      diag_name == "mVrVr_moment" || diag_name == "mVrVphi_moment" ||
      diag_name == "mVphiVphi_moment") {
      std::string moment_name =  //
        (diag_name == "density") ? "zeroth_moment" : diag_name;

      builder = std::make_unique<DistributionMomentBuilder>(
        simulation, result, moment_name, "(x_y_z)");

      PetscCall(builder->build(diag_info));
    }
    else {
      throw std::runtime_error("Unknown diagnostic name: " + diag_name);
    }
  }

  /// @todo Check uniqueness of result directories
  PetscFunctionReturn(PETSC_SUCCESS);
}
