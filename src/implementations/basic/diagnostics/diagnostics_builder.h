#ifndef SRC_BASIC_DIAGNOSTICS_DIAGNOSTICS_BUILDER_H
#define SRC_BASIC_DIAGNOSTICS_DIAGNOSTICS_BUILDER_H

#include "src/interfaces/diagnostic.h"

#include "src/pch.h"
#include "src/vectors/vector_classes.h"
#include "src/implementations/basic/simulation.h"
#include "src/implementations/basic/diagnostics/field_view.h"

namespace basic {

class Diagnostics_builder {
public:
  Diagnostics_builder(const Simulation& simulation);

  /// @todo change the signature to petsc-like
  std::vector<std::unique_ptr<interfaces::Diagnostic>> build();

private:
  const Simulation& simulation_;

  const Vec& get_field(const std::string& name) const;

  using Diagnostic_up = std::unique_ptr<interfaces::Diagnostic>;
  using Diagnostics_vector = std::vector<Diagnostic_up>;
  void build_fields_energy(const Configuration::json_t& diag_info, Diagnostics_vector& result);
  void build_fields_view(const Configuration::json_t& diag_info, Diagnostics_vector& result);
};

}

#endif  // SRC_BASIC_DIAGNOSTICS_DIAGNOSTICS_BUILDER_H
