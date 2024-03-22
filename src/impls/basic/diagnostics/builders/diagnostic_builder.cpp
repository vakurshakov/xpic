#include "diagnostic_builder.h"

#include "src/impls/basic/diagnostics/builders/fields_energy_builder.h"
#include "src/impls/basic/diagnostics/builders/field_view_builder.h"
#include "src/impls/basic/diagnostics/builders/distribution_moment_builder.h"


namespace basic {

Diagnostic_builder::Diagnostic_builder(const Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : simulation_(simulation), diagnostics_(diagnostics) {}


const Vec& Diagnostic_builder::get_field(const std::string& name) const {
  if (name == "E") return simulation_.E_;
  if (name == "B") return simulation_.B_;
  if (name == "J") return simulation_.J_;
  throw std::runtime_error("Unknown field name " + name);
}

Axis Diagnostic_builder::get_component(const std::string& name) const {
  if (name == "x") return X;
  if (name == "y") return Y;
  if (name == "z") return Z;
  throw std::runtime_error("Unknown component name " + name);
}

const Particles& Diagnostic_builder::get_sort(const std::string& name) const {
  const std::vector<Particles>& particles = simulation_.particles_;

  auto it = std::find_if(particles.begin(), particles.end(), [&](const Particles& sort) {
    return sort.parameters().sort_name == name;
  });

  if (it == particles.end()) {
    throw std::runtime_error("No particles with name " + name);
  }
  return *it;
}

Vector3<PetscReal> Diagnostic_builder::parse_vector(const Configuration::json_t& json, const std::string& name) const {
  std::string message;
  try {
    const Configuration::array_t& arr = json.at(name);

    if (arr.size() != 3) {
      message = name + " as array should be of size 3.";
      throw std::runtime_error(message);
    }

    Vector3<PetscReal> result;
    for (int i = 0; i < 3; ++i) {
      arr[i].get_to(result[i]);
    }
    return result;
  }
  catch (const std::exception& e) {
    message = e.what();
    message += usage_message();
    throw std::runtime_error(message);
  }
}


PetscErrorCode build_diagnostics(const Simulation& simulation, std::vector<Diagnostic_up>& result) {
  PetscFunctionBegin;

#if !THERE_ARE_FIELDS && !THERE_ARE_PARTICLES
  PetscFunctionReturn(PETSC_SUCCESS);
#endif

  LOG_TRACE("Building diagnostics");
  const Configuration& config = CONFIG();
  const Configuration::json_t& descriptions = config.json.at("Diagnostics");

  Diagnostic_builder_up builder = nullptr;

  for (const auto& [diag_name, diag_info] : descriptions.items()) {
#if THERE_ARE_FIELDS
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

#if THERE_ARE_PARTICLES
#if THERE_ARE_FIELDS
    else
#endif
    if (diag_name == "density"         ||
        diag_name == "Vx_moment"       ||
        diag_name == "Vy_moment"       ||
        diag_name == "Vz_moment"       ||
        diag_name == "mVxVx_moment"    ||
        diag_name == "mVxVy_moment"    ||
        diag_name == "mVxVz_moment"    ||
        diag_name == "mVyVy_moment"    ||
        diag_name == "mVyVz_moment"    ||
        diag_name == "mVzVz_moment"    ||
        diag_name == "Vr_moment"       ||
        diag_name == "Vphi_moment"     ||
        diag_name == "mVrVr_moment"    ||
        diag_name == "mVrVphi_moment"  ||
        diag_name == "mVphiVphi_moment") {
      LOG_INFO("Adding {} diagnostics(s)", diag_name);
      std::string moment_name = (diag_name == "density") ? "zeroth_moment" : diag_name;
      builder = std::make_unique<Distribution_moment_builder>(simulation, result, moment_name, "(x_y_z)");
      PetscCall(builder->build(diag_info));
    }
    else if (diag_name == "velocity") {
      LOG_INFO("Adding {} diagnostics(s)", diag_name);
      builder = std::make_unique<Distribution_moment_builder>(simulation, result, "zeroth_moment", "(Vx_Vy_Vz)");
      PetscCall(builder->build(diag_info));
    }
#endif
#if THERE_ARE_FIELDS || THERE_ARE_PARTICLES
    else
#endif
    {
      throw std::runtime_error("Unknown diagnostic name: " + diag_name);
    }
  }

  /// @todo Check uniqueness of result directories
  result.shrink_to_fit();
  PetscFunctionReturn(PETSC_SUCCESS);
}

}
