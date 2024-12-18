#ifndef SRC_BASIC_BUILDERS_DIAGNOSTIC_BUILDER_H
#define SRC_BASIC_BUILDERS_DIAGNOSTIC_BUILDER_H

#include "src/pch.h"
#include "src/interfaces/diagnostic.h"
#include "src/impls/basic/simulation.h"
#include "src/utils/configuration.h"


namespace basic {

class DiagnosticBuilder {
public:
  DEFAULT_MOVABLE(DiagnosticBuilder);

  DiagnosticBuilder(
    const Simulation& simulation, std::vector<Diagnostic_up>& diagnostics);
  virtual ~DiagnosticBuilder() = default;

  virtual PetscErrorCode build(const Configuration::json_t& diag_info) = 0;

protected:
  virtual std::string_view usage_message() const = 0;

  const Vec& get_field(const std::string& name) const;
  static Axis get_component(const std::string& name);

  const Particles& get_sort(const std::string& name) const;
  Vector3R parse_vector(
    const Configuration::json_t& json, const std::string& name) const;

  static PetscErrorCode check_region(
    const Vector3I& start, const Vector3I& size, const std::string& diag_name);

  const Simulation& simulation_;

  using Diagnostics_vector = std::vector<Diagnostic_up>;
  Diagnostics_vector& diagnostics_;
};

using Diagnostic_builder_up = std::unique_ptr<DiagnosticBuilder>;

PetscErrorCode build_diagnostics(
  const Simulation& simulation, std::vector<Diagnostic_up>& result);

}  // namespace basic

#endif  // SRC_BASIC_BUILDERS_DIAGNOSTIC_BUILDER_H
