#include "distribution_moment_builder.h"

#include "src/utils/utils.h"
#include "src/utils/vector_utils.h"

namespace basic {

Distribution_moment_builder::Distribution_moment_builder(
  const Simulation& simulation, std::vector<Diagnostic_up>& diagnostics,
  const std::string& moment_name, const std::string& proj_name)
  : Diagnostic_builder(simulation, diagnostics),
    moment_name(moment_name),
    proj_name(proj_name)
{
}


PetscErrorCode Distribution_moment_builder::build(
  const Configuration::json_t& diag_info)
{
  PetscFunctionBeginUser;

  auto parse_info = [&](const Configuration::json_t& info) -> PetscErrorCode {
    PetscFunctionBeginHot;
    Moment_description desc;
    PetscCall(parse_moment_info(info, desc));
    moments_desc_.emplace_back(std::move(desc));
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  /// @todo Vectorisation of parameters
  PetscCall(parse_info(diag_info));

  for (const Moment_description& desc : moments_desc_) {
    LOG("Add {}_of_{} diagnostic for {}", moment_name, proj_name, desc.particles_name);

    std::string res_dir = CONFIG().out_dir + "/" + desc.particles_name + "/" +
      moment_name + "_of_" + proj_name;

    const Particles& particles = get_sort(desc.particles_name);

    auto&& moment = Moment::from_string(particles, moment_name);

    if (auto&& diag = Distribution_moment::create(
          res_dir, particles, std::move(moment), desc.region)) {
      diagnostics_.emplace_back(std::move(diag));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Distribution_moment_builder::parse_moment_info(
  const Configuration::json_t& json, Moment_description& desc)
{
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

    PetscCall(check_region(vector_cast(desc.region.start), vector_cast(desc.region.size), desc.particles_name + " " + moment_name));
  }
  catch (const std::exception& e) {
    message = e.what();
    message += usage_message();
    throw std::runtime_error(message);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace basic
