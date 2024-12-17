#include "distribution_moment_builder.h"

#include "src/utils/utils.h"
#include "src/utils/vector_utils.h"

DistributionMomentBuilder::DistributionMomentBuilder(
  const interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics,
  const std::string& moment_name, const std::string& proj_name)
  : DiagnosticBuilder(simulation, diagnostics),
    moment_name(moment_name),
    proj_name(proj_name)
{
}


PetscErrorCode DistributionMomentBuilder::build(
  const Configuration::json_t& diag_info)
{
  PetscFunctionBeginUser;

  auto parse_info = [&](const Configuration::json_t& info) -> PetscErrorCode {
    PetscFunctionBeginHot;
    MomentDescription desc;
    PetscCall(parse_moment_info(info, desc));
    moments_desc_.emplace_back(std::move(desc));
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  /// @todo Vectorisation of parameters
  PetscCall(parse_info(diag_info));

  for (const MomentDescription& desc : moments_desc_) {
    LOG("Add {}_of_{} diagnostic for {}", moment_name, proj_name, desc.particles_name);

    std::string res_dir = CONFIG().out_dir + "/" + desc.particles_name + "/" +
      moment_name + "_of_" + proj_name;

    const basic::Particles& particles = get_sort(desc.particles_name);

    auto&& moment = Moment::from_string(particles, moment_name);

    if (auto&& diag =
          DistributionMoment::create(res_dir, particles, moment, desc.region)) {
      diagnostics_.emplace_back(std::move(diag));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DistributionMomentBuilder::parse_moment_info(
  const Configuration::json_t& json, MomentDescription& desc)
{
  PetscFunctionBeginUser;
  desc.region.dim = 3;
  desc.region.dof = 1;

  std::string message;
  try {
    json.at("sort").get_to(desc.particles_name);

    Vector3R start = parse_vector(json, "start");
    Vector3R size = parse_vector(json, "size");

    for (PetscInt i = 0; i < 3; ++i) {
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
