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
using Diagnostics_vector = std::vector<Diagnostic_up>;

Diagnostics_vector Diagnostics_builder::build() {
  Diagnostics_vector result;

#if INIT_CONFIGURATION
  LOG_TRACE("Building diagnostics");

  const Configuration& config = CONFIG();
  const Configuration::json_t& descriptions = config.json.at("Diagnostics");
#endif

  for (const auto& [diag_name, diag_info] : descriptions.items()) {
#if FIELDS_DIAGNOSTICS
    if (diag_name == "fields_energy") {
      LOG_INFO("Add fields energy diagnostic");
      build_fields_energy(diag_info, result);
    }
    else if (diag_name == "field_view") {
      LOG_INFO("Add field view diagnostic");
      build_field_view(diag_info, result);
    }
#endif
  }

  result.shrink_to_fit();
  return result;
}


void Diagnostics_builder::build_fields_energy(const Configuration::json_t& /* diag_info */, Diagnostics_vector& result) {
  result.emplace_back(std::make_unique<Fields_energy>(CONFIG().out_dir + "/", simulation_.da_, simulation_.E_, simulation_.B_));
}


Vec Diagnostics_builder::get_field(const std::string& name) const {
  if (name == "E") return simulation_.E_;
  if (name == "B") return simulation_.B_;
  throw std::runtime_error("Unknown field name!");
}

Axis Diagnostics_builder::get_component(const std::string& name) const {
  if (name == "x") return X;
  if (name == "y") return Y;
  if (name == "z") return Z;
  throw std::runtime_error("Unknown component name!");
}

struct Field_description {
  std::string field_name;
  std::string component_name;
};

using Field_descriptions = std::vector<Field_description>;

void from_json(const Configuration::json_t& json, Field_description& desc) {
  json.at("field").get_to(desc.field_name);
  json.at("comp").get_to(desc.component_name);
}

void Diagnostics_builder::build_field_view(const Configuration::json_t& diag_info, Diagnostics_vector& result) {
  Field_descriptions fields_desc;

  if (!diag_info.is_array()) {
    fields_desc.emplace_back(diag_info.get<Field_description>());
  }
  else {
    for (const Configuration::json_t& info : diag_info) {
      fields_desc.emplace_back(info.get<Field_description>());
    }
  }

  for (const auto& [field_name, component_name] : fields_desc) {
    LOG_INFO("Field view diagnostic is added for {}{}", field_name, component_name);

    result.emplace_back(std::make_unique<Field_view>(
      CONFIG().out_dir + "/" + field_name + component_name + "/",
      simulation_.da_, get_field(field_name), get_component(component_name)));
  }
}

}
