#include "diagnostic_builder.h"

#include "src/impls/basic/builders/distribution_moment_builder.h"
#include "src/impls/basic/builders/field_view_builder.h"
#include "src/impls/basic/builders/fields_energy_builder.h"
#include "src/utils/region_operations.h"


namespace basic {

DiagnosticBuilder::DiagnosticBuilder(
  const Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : simulation_(simulation), diagnostics_(diagnostics)
{
}


const Vec& DiagnosticBuilder::get_field(const std::string& name) const
{
  if (name == "E")
    return simulation_.E_;
  if (name == "B")
    return simulation_.B_;
  if (name == "J")
    return simulation_.J_;
  throw std::runtime_error("Unknown field name " + name);
}

/* static */ Axis DiagnosticBuilder::get_component(const std::string& name)
{
  if (name == "x")
    return X;
  if (name == "y")
    return Y;
  if (name == "z")
    return Z;
  throw std::runtime_error("Unknown component name " + name);
}


const Particles& DiagnosticBuilder::get_sort(const std::string& name) const
{
  const std::vector<Particles>& particles = simulation_.particles_;

  auto it =
    std::find_if(particles.begin(), particles.end(), [&](const Particles& sort) {
      return sort.parameters().sort_name == name;
    });

  if (it == particles.end())
    throw std::runtime_error("No particles with name " + name);
  return *it;
}


Vector3R DiagnosticBuilder::parse_vector(
  const Configuration::json_t& json, const std::string& name) const
{
  std::string message;
  try {
    const Configuration::array_t& arr = json.at(name);

    if (arr.size() != 3) {
      message = name + " as array should be of size 3.";
      throw std::runtime_error(message);
    }

    Vector3R result;
    for (PetscInt i = 0; i < 3; ++i)
      arr[i].get_to(result[i]);
    return result;
  }
  catch (const std::exception& e) {
    message = e.what();
    message += usage_message();
    throw std::runtime_error(message);
  }
}


/* static */ PetscErrorCode DiagnosticBuilder::check_region(
  const Vector3I& start, const Vector3I& size, const std::string& diag_name)
{
  PetscFunctionBeginUser;

  if (bool success = is_region_within_bounds(start, size, 0, Geom_n); !success) {
    throw std::runtime_error(
      "Region is not in global boundaries for " + diag_name + " diagnostic.");
  }

  if (bool success = (size[X] > 0) && (size[Y] > 0) && (size[Z] > 0); !success) {
    throw std::runtime_error(
      "Sizes are invalid for " + diag_name + " diagnostic.");
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode build_diagnostics(
  const Simulation& simulation, std::vector<Diagnostic_up>& result)
{
  PetscFunctionBeginUser;

  LOG("Building diagnostics");
  const Configuration& config = CONFIG();
  const Configuration::json_t& descriptions = config.json.at("Diagnostics");

  Diagnostic_builder_up builder = nullptr;

  for (const auto& [diag_name, diag_info] : descriptions.items()) {
    if (diag_name == "fields_energy") {
      LOG("Adding fields energy diagnostic");
      builder = std::make_unique<FieldsEnergyBuilder>(simulation, result);
      PetscCall(builder->build(diag_info));
    }
    else if (diag_name == "field_view") {
      LOG("Adding field view diagnostic(s)");
      builder = std::make_unique<FieldViewBuilder>(simulation, result);
      PetscCall(builder->build(diag_info));
    }
    else if (diag_name == "density" || diag_name == "Vx_moment" ||
      diag_name == "Vy_moment" || diag_name == "Vz_moment" ||
      diag_name == "mVxVx_moment" || diag_name == "mVxVy_moment" ||
      diag_name == "mVxVz_moment" || diag_name == "mVyVy_moment" ||
      diag_name == "mVyVz_moment" || diag_name == "mVzVz_moment" ||
      diag_name == "Vr_moment" || diag_name == "Vphi_moment" ||
      diag_name == "mVrVr_moment" || diag_name == "mVrVphi_moment" ||
      diag_name == "mVphiVphi_moment") {
      LOG("Adding {} diagnostics(s)", diag_name);
      std::string moment_name = (diag_name == "density") ? "zeroth_moment"
                                                         : diag_name;
      builder = std::make_unique<DistributionMomentBuilder>(
        simulation, result, moment_name, "(x_y_z)");
      PetscCall(builder->build(diag_info));
    }
    else {
      throw std::runtime_error("Unknown diagnostic name: " + diag_name);
    }
  }

  /// @todo Check uniqueness of result directories
  result.shrink_to_fit();
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace basic
