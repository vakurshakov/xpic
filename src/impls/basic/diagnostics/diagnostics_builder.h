#ifndef SRC_BASIC_DIAGNOSTICS_DIAGNOSTICS_BUILDER_H
#define SRC_BASIC_DIAGNOSTICS_DIAGNOSTICS_BUILDER_H

#include "src/interfaces/diagnostic.h"

#include "src/pch.h"
#include "src/vectors/vector_classes.h"
#include "src/impls/basic/simulation.h"
#include "src/impls/basic/diagnostics/field_view.h"

namespace basic {

class Diagnostics_builder {
public:
  Diagnostics_builder(const Simulation& simulation);

  PetscErrorCode build(std::vector<Diagnostic_up>& diagnostics);

private:
  const Simulation& simulation_;

  const Vec& get_field(const std::string& name) const;

  using Diagnostics_vector = std::vector<Diagnostic_up>;
  PetscErrorCode build_fields_energy(const Configuration::json_t& diag_info, Diagnostics_vector& result);
  PetscErrorCode build_fields_view(const Configuration::json_t& diag_info, Diagnostics_vector& result);
};

}

#endif  // SRC_BASIC_DIAGNOSTICS_DIAGNOSTICS_BUILDER_H
