#include "diagnostics_builder.h"

#include "src/implementations/basic/diagnostics/fields_energy.h"
#include "src/implementations/basic/diagnostics/field_view.h"

#define FIELDS_DIAGNOSTICS    (THERE_ARE_FIELDS && FIELDS_ARE_DIAGNOSED)
#define PARTICLES_DIAGNOSTICS (THERE_ARE_PARTICLES && PARTICLES_ARE_DIAGNOSED)
#define INIT_CONFIGURATION    (FIELDS_DIAGNOSTICS || PARTICLES_DIAGNOSTICS)

namespace basic {

Diagnostics_builder::Diagnostics_builder(const Simulation& simulation)
  : simulation_(simulation) {}

using Diagnostic_up = std::unique_ptr<interfaces::Diagnostic>;

std::vector<Diagnostic_up> Diagnostics_builder::build() {
  std::vector<Diagnostic_up> diagnostics;

#if INIT_CONFIGURATION
  LOG_TRACE("Building diagnostics");

  const Configuration& config = CONFIG();
  const auto& descriptions = config.get<Configuration::json>("Diagnostics");
#endif

  for (const auto& [name, description] : descriptions.items()) {
#if FIELDS_DIAGNOSTICS
    if (name == "fields_energy") {
      LOG_INFO("Add fields energy diagnostic");
      diagnostics.emplace_back(build_fields_energy(description));
    }
    else if (name == "field_view") {
      LOG_INFO("Add field view diagnostic");
      diagnostics.emplace_back(build_field_view(description));
    }
#endif
  }

  diagnostics.shrink_to_fit();
  return diagnostics;
}


Diagnostic_up Diagnostics_builder::build_fields_energy(const Configuration::json&) {
  const Configuration& config = CONFIG();
  return std::make_unique<Fields_energy>(config.out_dir + "/", simulation_.da_, simulation_.E_, simulation_.B_);
}


Vec Diagnostics_builder::get_field(const std::string& name) const {
  if (name == "E") return simulation_.E_;
  if (name == "B") return simulation_.B_;
  throw std::runtime_error("Unknown field name!");
}

Diagnostic_up Diagnostics_builder::build_field_view(const Configuration::json& description) {
  const Configuration& config = CONFIG();

  // config :)))
  std::string field_name = "B";
  return std::make_unique<Field_view>(config.out_dir + "/" + field_name + "/", simulation_.da_, get_field(field_name));
}

}
