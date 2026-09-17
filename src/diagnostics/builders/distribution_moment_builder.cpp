#include "distribution_moment_builder.h"

#include "src/diagnostics/distribution_moment.h"
#include "src/utils/utils.h"
#include "src/utils/vector_utils.h"
#include "src/impls/drift_kinetic/particles.h"
#include "src/impls/drift_kinetic/diagnostic.h"

DistributionMomentBuilder::DistributionMomentBuilder(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : FieldViewBuilder(simulation, diagnostics)
{
}

PetscErrorCode DistributionMomentBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  const std::map<std::string_view, PetscInt> available_moments{
    {"density", 1},
    {"current", 3},
    {"velocity", 3},
    {"velocity_parallel", 1},
    {"temperature_perp", 1},
    {"temperature_parallel", 1},
    {"momentum_flux", 6},
    {"momentum_flux_cyl", 6},
    {"momentum_flux_diag", 3},
    {"momentum_flux_diag_cyl", 3},
  };

  std::string particles;
  info.at("particles").get_to(particles);

  std::string moment;
  info.at("moment").get_to(moment);

  auto it = available_moments.find(moment);
  if (it == available_moments.end())
    throw std::runtime_error(
      "Unknown moment name " + moment + " for particles " + particles);

  std::string out_dir = particles + "/" + moment;

  Region region;
  region.dim = it->second > 1 ? 4 : 3;
  region.dof = it->second;
  region.start = Vector4I(0);
  region.size = Vector4I(geom_nx, geom_ny, geom_nz, region.dof);

  if (info.contains("region"))
    parse_region(info.at("region"), region, particles + " " + moment);

  if (info.contains("out_dir"))
    info.at("out_dir").get_to(out_dir);

  LOG("  {} diagnostic is added for {}, output directory: {}", moment, particles, out_dir);

  std::unique_ptr<DistributionMoment> diagnostic;

  /// @todo Unify this code to avoid separation by different particle types
  auto* dk_particles = dynamic_cast<const drift_kinetic::Particles*>(&simulation_.get_named_particles(particles));

  if (dk_particles != nullptr) {
    if (auto dk_m = drift_kinetic::dk_moment_from_string(moment); dk_m != nullptr) {
      diagnostic = drift_kinetic::DkDistributionMoment::create(
        out_dir, *dk_particles, dk_m, region);
    } else {
      diagnostic = drift_kinetic::DkDistributionMoment::create(
        out_dir, *dk_particles, moment_from_string(moment), region);
    }
  } else {
    diagnostic = DistributionMoment::create(
      out_dir, simulation_.get_named_particles(particles),
      moment_from_string(moment), region);
  }

  if (!diagnostic)
    PetscFunctionReturn(PETSC_SUCCESS);

  diagnostics_.emplace_back(std::move(diagnostic));
  PetscFunctionReturn(PETSC_SUCCESS);
}
