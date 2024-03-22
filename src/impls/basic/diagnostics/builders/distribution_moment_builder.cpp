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
    PetscFunctionBegin;
    Moment_description desc;
    PetscCall(parse_moment_info(info, desc));
    moments_desc_.emplace_back(std::move(desc));
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  // if (!diag_info.is_array())
  PetscCall(parse_info(diag_info));

  for (const Moment_description& desc : moments_desc_) {
    LOG_INFO("Add {}_of_{} diagnostic for {}", moment_name, proj_name, desc.particles_name);

    std::string res_dir = CONFIG().out_dir + "/" + moment_name + "_of_" + proj_name;

    const Particles& particles = get_sort(desc.particles_name);

    std::unique_ptr<Distribution_moment>&& diag = std::make_unique<Distribution_moment>(
      res_dir, simulation_.da_, particles,
      std::make_unique<Moment>(particles, moment_name),
      std::make_unique<Projector>(particles, proj_name));

    PetscCall(diag->set_diagnosed_region(desc.region));

    diagnostics_.emplace_back(std::move(diag));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Distribution_moment_builder::parse_moment_info(const Configuration::json_t& json, Moment_description& desc) {
  PetscFunctionBegin;
  std::string message;
  try {
    json.at("sort").get_to(desc.particles_name);

    Vector3<PetscReal> start = parse_vector(json, "start");
    Vector3<PetscReal> size = parse_vector(json, "size");
    desc.region.dp = parse_vector(json, "dp");

    for (int i = 0; i < 3; ++i) {
      desc.region.start[i] = TO_STEP(start[i], desc.region.dp[i]);
      desc.region.size[i] = TO_STEP(size[i], desc.region.dp[i]);
    }
  }
  catch (const std::exception& e) {
    message = e.what();
    message += usage_message();
    throw std::runtime_error(message);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}
