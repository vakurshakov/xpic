#include "field_view_builder.h"

#include "src/utils/region_operations.h"
#include "src/vectors/vector_utils.h"

namespace basic {

Field_view_builder::Field_view_builder(const Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : Diagnostic_builder(simulation, diagnostics) {}

PetscErrorCode Field_view_builder::build(const Configuration::json_t& diag_info) {
  PetscFunctionBeginUser;

  auto parse_info = [&](const Configuration::json_t& info) -> PetscErrorCode {
    PetscFunctionBeginUser;
    Field_description desc;
    PetscCall(parse_field_info(info, desc));
    PetscCall(check_field_description(desc));
    fields_desc_.emplace_back(std::move(desc));
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  if (!diag_info.is_array()) {
    PetscCall(parse_info(diag_info));
  }
  else {
    for (const Configuration::json_t& info : diag_info) {
      PetscCall(parse_info(info));
    }
  }

  for (const Field_description& desc : fields_desc_) {
    LOG_INFO("Field view diagnostic is added for {}{}", desc.field_name, desc.component_name);

    std::string res_dir = CONFIG().out_dir + "/" + desc.field_name + desc.component_name + "/";

    if (auto&& diag = Field_view::create(res_dir, simulation_.da_, get_field(desc.field_name), desc.region)) {
      diagnostics_.emplace_back(std::move(diag));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Field_view_builder::parse_field_info(const Configuration::json_t& json, Field_description& desc) {
  PetscFunctionBeginUser;
  desc.region.dim = 4;
  desc.region.dof = 3;

  std::string message;
  try {
    json.at("field").get_to(desc.field_name);

    desc.region.start[3] = 0;
    desc.region.size[3] = 3;

    if (json.contains("comp")) {
      json.at("comp").get_to(desc.component_name);
      if (!desc.component_name.empty()) {
        desc.region.start[3] = get_component(desc.component_name);
        desc.region.size[3] = 1;
      }
    }

    Vector3R start = parse_vector(json, "start");
    Vector3R size = parse_vector(json, "size");

    for (int i = 0; i < 3; ++i) {
      desc.region.start[i] = TO_STEP(start[i], Dx[i]);
      desc.region.size[i] = TO_STEP(size[i], Dx[i]);
    }
  }
  catch (const std::exception& e) {
    message = e.what();
    message += usage_message();
    throw std::runtime_error(message);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @todo Move check into base class
PetscErrorCode Field_view_builder::check_field_description(const Field_description& desc) {
  PetscFunctionBeginUser;
  std::string message;

  Vector3I r_start = vector_cast(desc.region.start);
  Vector3I r_size = vector_cast(desc.region.size);
  bool is_region_in_global_bounds = is_region_within_bounds(r_start, r_size, 0, Geom_n);
  message = "Region is not in global boundaries for " + desc.field_name + desc.component_name + " diagnostic.";
  PetscCheck(is_region_in_global_bounds, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, message.c_str());

  bool are_sizes_positive = (r_size[X] > 0) && (r_size[Y] > 0) && (r_size[Z] > 0);
  message = "Sizes are invalid for " + desc.field_name + desc.component_name + " diagnostic.";
  PetscCheck(are_sizes_positive, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, message.c_str());

  PetscFunctionReturn(PETSC_SUCCESS);
}

}
