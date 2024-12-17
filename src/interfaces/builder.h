#ifndef SRC_INTERFACES_BUILDER_H
#define SRC_INTERFACES_BUILDER_H

#include "src/pch.h"
#include "src/impls/basic/simulation.h"
#include "src/utils/configuration.h"


namespace interfaces {

class Builder {
public:
  DEFAULT_MOVABLE(Builder);

  Builder(const Simulation& simulation);
  virtual ~Builder() = default;

  virtual PetscErrorCode build(const Configuration::json_t& diag_info) = 0;

protected:
  virtual std::string_view usage_message() const = 0;

  static Axis get_component(const std::string& name);

  Vec get_global_vector(const std::string& name) const;

  Vector3R parse_vector(
    const Configuration::json_t& json, const std::string& name) const;

  static PetscErrorCode check_region(
    const Vector3I& start, const Vector3I& size, const std::string& diag_name);

  const Simulation& simulation_;
};

}  // namespace interfaces

using Builder_up = std::unique_ptr<interfaces::Builder>;

#endif  // SRC_INTERFACES_BUILDER_H
