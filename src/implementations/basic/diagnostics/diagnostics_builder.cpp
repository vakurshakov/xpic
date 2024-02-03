#include "diagnostics_builder.h"

#include "src/implementations/basic/diagnostics/fields_energy.h"

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
      diagnostics.emplace_back(build_fields_energy());
    }
#endif
  }

  diagnostics.shrink_to_fit();
  return diagnostics;
}


Diagnostic_up Diagnostics_builder::build_fields_energy() {
  const Configuration& config = CONFIG();
  return std::make_unique<Fields_energy>(config.out_dir + "/", simulation_.da_, simulation_.E_, simulation_.B_);
}

}
