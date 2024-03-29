#ifndef SRC_BASIC_DIAGNOSTICS_BUILDERS_DIAGNOSTIC_BUILDER_H
#define SRC_BASIC_DIAGNOSTICS_BUILDERS_DIAGNOSTIC_BUILDER_H

#include "src/interfaces/diagnostic.h"

#include "src/pch.h"
#include "src/impls/basic/simulation.h"
#include "src/vectors/vector_classes.h"
#include "src/utils/configuration.h"

namespace basic {

class Diagnostic_builder {
public:
  Diagnostic_builder(const Simulation& simulation, std::vector<Diagnostic_up>& diagnostics);
  virtual ~Diagnostic_builder() = default;

  virtual PetscErrorCode build(const Configuration::json_t& diag_info) = 0;

protected:
  virtual constexpr std::string usage_message() const = 0;

  const Vec& get_field(const std::string& name) const;
  Axis get_component(const std::string& name) const;

  const Particles& get_sort(const std::string& name) const;
  Vector3<PetscReal> parse_vector(const Configuration::json_t& json, const std::string& name) const;

  bool is_region_within_bounds(
    const Vector3<PetscInt>& r_start, const Vector3<PetscInt>& r_size,
    const Vector3<PetscInt>& b_start, const Vector3<PetscInt>& b_size) const;

  bool is_region_intersect_bounds(
    const Vector3<PetscInt>& r_start, const Vector3<PetscInt>& r_size,
    const Vector3<PetscInt>& b_start, const Vector3<PetscInt>& b_size) const;

protected:
  const Simulation& simulation_;

  using Diagnostics_vector = std::vector<Diagnostic_up>;
  Diagnostics_vector& diagnostics_;

};

using Diagnostic_builder_up = std::unique_ptr<Diagnostic_builder>;

PetscErrorCode build_diagnostics(const Simulation& simulation, std::vector<Diagnostic_up>& diagnostics);

}

#endif  // SRC_BASIC_DIAGNOSTICS_BUILDERS_DIAGNOSTIC_BUILDER_H
