#include "distribution_moment_builder.h"

#include "src/utils/utils.h"
#include "src/utils/vector_utils.h"

DistributionMomentBuilder::DistributionMomentBuilder(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : FieldViewBuilder(simulation, diagnostics)
{
}

PetscErrorCode DistributionMomentBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
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

  std::string particles;
  info.at("particles").get_to(particles);

  std::string moment;
  info.at("moment").get_to(moment);

  if (!available_moments.contains(moment))
    throw std::runtime_error(
      "Unknown moment name " + moment + " for particles " + particles);

  FieldView::Region region;
  region.start = Vector4I(0);
  region.size = Vector4I(geom_nx, geom_ny, geom_nz, 1);
  region.dim = 3;
  region.dof = 1;

  std::string suffix;

  if (auto it = info.find("region"); it != info.end()) {
    parse_region_start_size(*it, region, particles + " " + moment);
    parse_res_dir_suffix(*it, suffix);
  }

  check_region(region, particles + " " + moment);

  LOG("  {} diagnostic is added for {}, suffix: {}", moment, particles, suffix.empty() ? "<empty>" : suffix);

  std::string res_dir =
    CONFIG().out_dir + "/" + particles + "/" + moment + suffix;

  auto&& diagnostic = DistributionMoment::create(res_dir,
    simulation_.get_named_particles(particles), moment_from_string(moment),
    region);

  if (!diagnostic)
    PetscFunctionReturn(PETSC_SUCCESS);

  diagnostics_.emplace_back(std::move(diagnostic));
  PetscFunctionReturn(PETSC_SUCCESS);
}
