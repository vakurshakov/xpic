#include "distribution_moment_builder.h"

namespace basic {

Distribution_moment_builder::Distribution_moment_builder(
  const Simulation& simulation, std::vector<Diagnostic_up>& diagnostics,
  const std::string& moment_name, const std::string& proj_name)
  : Diagnostic_builder(simulation, diagnostics),
    moment_name(moment_name), proj_name(proj_name) {}


PetscErrorCode Distribution_moment_builder::build(const Configuration::json_t& diag_info) {
  PetscFunctionBeginUser;

  auto parse_info = [&](const Configuration::json_t& info) -> PetscErrorCode {
    PetscFunctionBeginHot;
    Moment_description desc;
    PetscCall(parse_moment_info(info, desc));
    PetscCall(check_moment_description(desc));
    moments_desc_.emplace_back(std::move(desc));
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  /// @todo Vectorisation of parameters
  PetscCall(parse_info(diag_info));

  for (const Moment_description& desc : moments_desc_) {
    LOG_INFO("Add {}_of_{} diagnostic for {}", moment_name, proj_name, desc.particles_name);

    std::string res_dir = CONFIG().out_dir + "/" + desc.particles_name + "/" + moment_name + "_of_" + proj_name;

    const Particles& particles = get_sort(desc.particles_name);

    auto&& moment = std::make_unique<Moment>(particles, moment_name);

    if (auto&& diag = Distribution_moment::create(res_dir, simulation_.da_, particles, std::move(moment), desc.region)) {
      diagnostics_.emplace_back(std::move(diag));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Distribution_moment_builder::parse_moment_info(const Configuration::json_t& json, Moment_description& desc) {
  PetscFunctionBeginUser;
  desc.region.dim = 3;
  desc.region.dof = 1;

  std::string message;
  try {
    json.at("sort").get_to(desc.particles_name);

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

/// @todo We can move check and attach into base class too
PetscErrorCode Distribution_moment_builder::check_moment_description(const Moment_description& desc) {
  PetscFunctionBeginUser;
  std::string message;

  Vector3I r_start(desc.region.start);
  Vector3I r_size(desc.region.size);
  bool is_region_in_global_bounds = is_region_within_bounds(r_start, r_size, 0, Geom_n);
  message = "Region is not in global boundaries for " + moment_name + " diagnostic.";
  PetscCheck(is_region_in_global_bounds, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, message.c_str());

  bool are_sizes_positive = (r_size[X] > 0) && (r_size[Y] > 0) && (r_size[Z] > 0);
  message = "Sizes are invalid for " + moment_name + " diagnostic.";
  PetscCheck(are_sizes_positive, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, message.c_str());

  PetscFunctionReturn(PETSC_SUCCESS);
}

}
