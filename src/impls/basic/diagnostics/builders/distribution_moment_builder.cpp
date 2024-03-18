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

    const Particles& particles = find_sort(desc.particles_name);

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

    /// @todo Move common parts into the base class
    const Configuration::array_t& start = json.at("start");
    const Configuration::array_t& size = json.at("size");
    const Configuration::array_t& dp = json.at("dp");

    bool sizes_are_correct = (start.size() == 3) && (size.size() == 3) && (dp.size() == 3);
    message = "Start, size and dp as arrays should be of size 3, representing all of 3 dimensions.";
    PetscCheck(sizes_are_correct, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, message.c_str());

    // Region in the configuration file is in global coordinates, in c/w_pe units
    for (int i = 0; i < 3; ++i) {
      desc.region.start[i] = TO_STEP(start[i].get<PetscReal>(), Dx[i]);
      desc.region.size[i] = TO_STEP(size[i].get<PetscReal>(), Dx[i]);
      desc.region.dp[i] = dp[i].get<PetscReal>();
    }
  }
  catch (const std::exception& e) {
    message = e.what();
    message += usage_message();
    throw std::runtime_error(message);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

const Particles& Distribution_moment_builder::find_sort(const std::string& sort_name) const {
  const std::vector<Particles>& particles = simulation_.particles_;

  auto it = std::find_if(particles.begin(), particles.end(), [&](const Particles& sort) {
    return sort.parameters().sort_name == sort_name;
  });

  if (it == particles.end()) {
    throw std::runtime_error("No particles with name " + sort_name);
  }
  return *it;
}

}
