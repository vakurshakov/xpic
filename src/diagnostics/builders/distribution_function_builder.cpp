#include "distribution_function_builder.h"

#include "src/diagnostics/distribution_function.h"
#include "src/impls/drift_kinetic/diagnostic.h"
#include "src/impls/drift_kinetic/particles.h"
#include "src/utils/utils.h"

DistributionFunctionBuilder::DistributionFunctionBuilder(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : DistributionMomentBuilder(simulation, diagnostics)
{
}

PetscErrorCode DistributionFunctionBuilder::build(
  const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  std::string particles_name;
  info.at("particles").get_to(particles_name);

  auto& particles = simulation_.get_named_particles(particles_name);
  if (auto* dk_particles = dynamic_cast<drift_kinetic::Particles*>(&particles)) {
    PetscCall(build_dk_distribution_function(info));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  std::string projector_name;
  info.at("projector").get_to(projector_name);

  const auto& geometry = info.at("geometry");

  std::string geometry_name;
  geometry.at("name").get_to(geometry_name);

  Tester test;
  Region xreg_aabb;
  xreg_aabb.dim = 2;
  xreg_aabb.dof = 1;

  if (geometry_name == "BoxGeometry") {
    BoxGeometry box;
    load_geometry(geometry, box);
    test = WithinBox(box);

    xreg_aabb.start = Vector4I{
      FLOOR_STEP(box.min[X], dx),
      FLOOR_STEP(box.min[Y], dy),
      FLOOR_STEP(box.min[Z], dz),
      0,
    };

    xreg_aabb.size = Vector4I{
      FLOOR_STEP(box.max[X], dx),
      FLOOR_STEP(box.max[Y], dy),
      FLOOR_STEP(box.max[Z], dz),
      1,
    };

    xreg_aabb.size -= xreg_aabb.start;
  }
  else if (geometry_name == "CylinderGeometry") {
    CylinderGeometry cyl;
    load_geometry(geometry, cyl);
    test = WithinCylinder(cyl);

    xreg_aabb.start = Vector4I{
      FLOOR_STEP(cyl.center[X] - cyl.radius, dx),
      FLOOR_STEP(cyl.center[Y] - cyl.radius, dy),
      FLOOR_STEP(cyl.center[Z] - 0.5 * cyl.height, dz),
      0,
    };

    xreg_aabb.size = Vector4I{
      FLOOR_STEP(cyl.center[X] + cyl.radius, dx),
      FLOOR_STEP(cyl.center[Y] + cyl.radius, dy),
      FLOOR_STEP(cyl.center[Z] + 0.5 * cyl.height, dz),
      1,
    };

    xreg_aabb.size -= xreg_aabb.start;
  }
  else {
    throw std::runtime_error("Unknown geometry name " + geometry_name);
  }

  DistributionFunction::VelocityRegion vreg{
    .vx_max = +1,
    .vy_max = +1,
    .vx_min = -1,
    .vy_min = -1,
  };

  const auto& dv = info.at("dv");
  dv[0].get_to(vreg.dvx);
  dv[1].get_to(vreg.dvy);

  auto it = info.find("vmax");
  if (it != info.end()) {
    vreg.vx_max = (*it)[0];
    vreg.vy_max = (*it)[1];
  }

  it = info.find("vmin");
  if (it != info.end()) {
    vreg.vx_min = (*it)[0];
    vreg.vy_min = (*it)[1];
  }

  std::string out_dir = particles_name + "/" + projector_name;

  if (info.contains("out_dir"))
    info.at("out_dir").get_to(out_dir);

  out_dir = CONFIG().out_dir + "/" + out_dir;

  LOG("  {} velocity distribution diagnostic is added for {}, output directory: {}", projector_name, particles_name, out_dir);

  auto&& diagnostic = DistributionFunction::create(out_dir, particles,
    projector_from_string(projector_name), test, xreg_aabb, vreg);

  if (diagnostic)
    diagnostics_.emplace_back(std::move(diagnostic));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @todo `DkDistributionFunction` is internally inherits the logic from `DistributionFunction`
PetscErrorCode DistributionFunctionBuilder::build_dk_distribution_function(
  const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  std::string particles_name;
  info.at("particles").get_to(particles_name);

  auto* particles = &simulation_.get_named_particles(particles_name);
  auto* dk_particles = dynamic_cast<drift_kinetic::Particles*>(particles);

  auto parse_dk_phase_axis =
    [](const Configuration::json_t& info, const std::string& name)
  {
    auto axis_info = info.at(name);

    drift_kinetic::DkPhaseAxis axis{
      .min = parse_value(axis_info.at("min")),
      .max = parse_value(axis_info.at("max")),
      .bins = axis_info.at("bins").get<PetscInt>(),
    };

    if (!(axis.min < axis.max && axis.bins > 0))
      throw std::runtime_error(
        "DkDistributionFunction requires a non-empty"
        "range and positive bins for " +
        name);
    return axis;
  };

  auto v_par = parse_dk_phase_axis(info, "v_parallel");
  auto mu_p = parse_dk_phase_axis(info, "mu_p");

  auto dp_parsed = info.value("diagnose_period", diagnose_period);
  auto diagnose_period = std::max(1, ROUND_STEP(parse_value(dp_parsed), dt));
  if (diagnose_period <= 0)
    throw std::runtime_error(
      "DkDistributionFunction diagnose_period must be positive");

  auto max_frames = info.value("max_frames", (PetscInt)(-1));
  if (info.contains("max_frames") && max_frames <= 0)
    throw std::runtime_error(
      "DkDistributionFunction max_frames must be positive or omitted");

  std::string out_dir = particles_name + "/distribution_function";

  if (info.contains("out_dir"))
    info.at("out_dir").get_to(out_dir);

  out_dir = CONFIG().out_dir + "/" + out_dir;

  LOG("  DkDistributionFunction diagnostic is added for {}, output directory: {}", particles_name, out_dir);

  diagnostics_.emplace_back(drift_kinetic::DkDistributionFunction::create(
    out_dir, *dk_particles, v_par, mu_p, diagnose_period, max_frames));
  PetscFunctionReturn(PETSC_SUCCESS);
}
