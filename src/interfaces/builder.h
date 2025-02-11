#ifndef SRC_INTERFACES_BUILDER_H
#define SRC_INTERFACES_BUILDER_H

#include "src/pch.h"
#include "src/impls/basic/simulation.h"
#include "src/utils/configuration.h"
#include "src/utils/geometries.h"


namespace interfaces {

class Builder {
public:
  DEFAULT_MOVABLE(Builder);

  Builder(const Simulation& simulation);
  virtual ~Builder() = default;

  virtual PetscErrorCode build(const Configuration::json_t& info) = 0;

  template<class InheritedBuilder, class Container>
  static PetscErrorCode use_impl(const Configuration::json_t& info,
    const interfaces::Simulation& simulation, Container& result)
  {
    PetscFunctionBeginUser;
    auto&& builder = std::make_unique<InheritedBuilder>(simulation, result);
    try {
      PetscCall(builder->build(info));
    }
    catch (const std::exception& e) {
      std::string message;
      message = e.what();
      message += builder->usage_message();
      throw std::runtime_error(message);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  virtual std::string_view usage_message() const = 0;

protected:
  static Axis get_component(const std::string& name);

  Vector3R parse_vector(
    const Configuration::json_t& info, const std::string& name) const;

  void check_region(
    const Vector3I& start, const Vector3I& size, const std::string& name) const;

  void load_geometry(const Configuration::json_t& info, BoxGeometry& box);
  void load_geometry(const Configuration::json_t& info, CylinderGeometry& cyl);

  Simulation& simulation_;
};

}  // namespace interfaces

using Builder_up = std::unique_ptr<interfaces::Builder>;

#endif  // SRC_INTERFACES_BUILDER_H
