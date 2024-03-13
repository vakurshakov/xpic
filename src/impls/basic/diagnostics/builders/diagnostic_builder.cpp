#include "diagnostic_builder.h"

#include "src/impls/basic/diagnostics/builders/fields_energy_builder.h"
#include "src/impls/basic/diagnostics/builders/field_view_builder.h"
#include "src/impls/basic/diagnostics/builders/distribution_moment_builder.h"

#define FIELDS_DIAGNOSTICS    (THERE_ARE_FIELDS && FIELDS_ARE_DIAGNOSED)
#define PARTICLES_DIAGNOSTICS (THERE_ARE_PARTICLES && PARTICLES_ARE_DIAGNOSED)


namespace basic {

using Diagnostics_vector = std::vector<Diagnostic_up>;

Diagnostic_builder::Diagnostic_builder(const Simulation& simulation, Diagnostics_vector& diagnostics)
  : simulation_(simulation), diagnostics_(diagnostics) {}


const Vec& Diagnostic_builder::get_field(const std::string& name) const {
  if (name == "E") return simulation_.E_;
  if (name == "B") return simulation_.B_;
  throw std::runtime_error("Unknown field name " + name);
}

Axis Diagnostic_builder::get_component(const std::string& name) const {
  if (name == "x") return X;
  if (name == "y") return Y;
  if (name == "z") return Z;
  throw std::runtime_error("Unknown component name " + name);
}


PetscErrorCode build_diagnostics(const Simulation& simulation, Diagnostics_vector& result) {
  PetscFunctionBegin;

#if !FIELDS_DIAGNOSTICS && !PARTICLES_DIAGNOSTICS
  PetscFunctionReturn(PETSC_SUCCESS);
#endif

  LOG_TRACE("Building diagnostics");
  const Configuration& config = CONFIG();
  const Configuration::json_t& descriptions = config.json.at("Diagnostics");

  Diagnostic_builder_up builder = nullptr;

  for (const auto& [diag_name, diag_info] : descriptions.items()) {
#if FIELDS_DIAGNOSTICS
    if (diag_name == "fields_energy") {
      LOG_INFO("Adding fields energy diagnostic");
      builder = std::make_unique<Fields_energy_builder>(simulation, result);
      PetscCall(builder->build(diag_info));
    }
    else if (diag_name == "field_view") {
      LOG_INFO("Adding field view diagnostic(s)");
      builder = std::make_unique<Field_view_builder>(simulation, result);
      PetscCall(builder->build(diag_info));
    }
#endif

#if PARTICLES_DIAGNOSTICS
    if (diag_name == "density") {
      LOG_INFO("Adding density diagnostic(s)");
      builder = std::make_unique<Distribution_moment_builder>(simulation, result, "zeroth_moment", "XY");
      PetscCall(builder->build(diag_info));
    }
#endif
    /// @todo Implement exception on unkown diagnostic
  }

  /// @todo Check uniqueness of result directories
  result.shrink_to_fit();
  PetscFunctionReturn(PETSC_SUCCESS);
}

}
