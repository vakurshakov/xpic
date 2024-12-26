#include "distribution_moment_builder.h"

#include <set>

#include "src/utils/utils.h"
#include "src/utils/vector_utils.h"

DistributionMomentBuilder::DistributionMomentBuilder(
  const interfaces::Simulation& simulation,
  std::vector<Diagnostic_up>& diagnostics)
  : DiagnosticBuilder(simulation, diagnostics)
{
}

PetscErrorCode DistributionMomentBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;

  FieldView::Region region;
  region.dim = 3;
  region.dof = 1;

  std::string particles;
  std::string moment;

  std::set<std::string_view> available_moments{
    "Density",
    "Vx",
    "Vy",
    "Vz",
    "Vr",
    "Vphi",
    "mVxVx",
    "mVxVy",
    "mVxVz",
    "mVyVy",
    "mVyVz",
    "mVzVz",
    "mVrVr",
    "mVrVphi",
    "mVrVz",
    "mVphiVphi",
    "mVphiVz",
  };

  std::string message;
  try {
    info.at("particles").get_to(particles);
    info.at("moment").get_to(moment);

    if (!available_moments.contains(moment))
      throw std::runtime_error(
        "Unknown moment name " + moment + " for particles " + particles);

    Vector3R start{0, 0, 0};
    Vector3R size{geom_x, geom_y, geom_z};

    if (info.contains("start"))
      start = parse_vector(info, "start");

    if (info.contains("size"))
      size = parse_vector(info, "size");

    for (PetscInt i = 0; i < 3; ++i) {
      region.start[i] = TO_STEP(start[i], Dx[i]);
      region.size[i] = TO_STEP(size[i], Dx[i]);
    }

    check_region(vector_cast(region.start), vector_cast(region.size),
      particles + " " + moment);
  }
  catch (const std::exception& e) {
    message = e.what();
    message += usage_message();
    throw std::runtime_error(message);
  }

  LOG("{} diagnostic is added for {}", moment, particles);

  std::string res_dir = CONFIG().out_dir + "/" + particles + "/" + moment;

  auto&& diagnostic = DistributionMoment::create(res_dir,
    simulation_.get_named_particles(particles), moment_from_string(moment),
    region);

  if (!diagnostic)
    PetscFunctionReturn(PETSC_SUCCESS);

  diagnostics_.emplace_back(std::move(diagnostic));
  PetscFunctionReturn(PETSC_SUCCESS);
}
