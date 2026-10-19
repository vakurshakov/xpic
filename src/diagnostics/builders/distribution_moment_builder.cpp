#include "distribution_moment_builder.h"

#include "src/diagnostics/distribution_moment.h"
#include "src/impls/drift_kinetic/diagnostic.h"
#include "src/impls/drift_kinetic/particles.h"
#include "src/utils/utils.h"

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

  std::string particles_name;
  info.at("particles").get_to(particles_name);

  std::string moment_name;
  info.at("moment").get_to(moment_name);

  auto it = available_moments.find(moment_name);
  if (it == available_moments.end())
    throw std::runtime_error(
      "Unknown moment name " + moment_name + " for particles " + particles_name);

  std::string out_dir = particles_name + "/" + moment_name;

  Region region;
  region.dim = it->second > 1 ? 4 : 3;
  region.dof = it->second;
  region.start = Vector4I(0);
  region.size = Vector4I(geom_nx, geom_ny, geom_nz, region.dof);

  if (info.contains("region"))
    parse_region(info.at("region"), region, particles_name + " " + moment_name);

  if (info.contains("out_dir"))
    info.at("out_dir").get_to(out_dir);

  LOG("  {} diagnostic is added for {}, output directory: {}", moment_name, particles_name, out_dir);

  out_dir = CONFIG().out_dir + "/" + out_dir;

  std::unique_ptr<DistributionMoment> diagnostic;

  auto& particles = simulation_.get_named_particles(particles_name);

  /// @todo Unify this code to avoid separation by different particle types
  if (auto* dk_particles = dynamic_cast<drift_kinetic::Particles*>(&particles)) {
    auto dk_moment = drift_kinetic::dk_moment_from_string(moment_name);
    diagnostic = dk_moment  //
      ? drift_kinetic::DkDistributionMoment::create(
          out_dir, *dk_particles, dk_moment, region)
      : drift_kinetic::DkDistributionMoment::create(
          out_dir, *dk_particles, moment_from_string(moment_name), region);
  }
  else {
    diagnostic = DistributionMoment::create(
      out_dir, particles, moment_from_string(moment_name), region);
  }

  if (diagnostic)
    diagnostics_.emplace_back(std::move(diagnostic));
  PetscFunctionReturn(PETSC_SUCCESS);
}
