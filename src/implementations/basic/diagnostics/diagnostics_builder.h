#ifndef SRC_BASIC_DIAGNOSTICS_DIAGNOSTICS_BUILDER_H
#define SRC_BASIC_DIAGNOSTICS_DIAGNOSTICS_BUILDER_H

#include "src/interfaces/diagnostic.h"

#include "src/pch.h"
#include "src/implementations/basic/simulation.h"

namespace basic
{

class Diagnostics_builder {
public:
  Diagnostics_builder(const Simulation& simulation);
  std::vector<std::unique_ptr<interfaces::Diagnostic>> build();

private:
  const Simulation& simulation_;

  Vec get_field(const std::string& name) const;

  using Diagnostic_up = std::unique_ptr<interfaces::Diagnostic>;
  Diagnostic_up build_fields_energy(const Configuration::json_t& description);
  Diagnostic_up build_field_view(const Configuration::json_t& description);
};

}

#endif  // SRC_BASIC_DIAGNOSTICS_DIAGNOSTICS_BUILDER_H
