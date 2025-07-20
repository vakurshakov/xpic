#include "velocity_distribution_builder.h"

#include "src/diagnostics/velocity_distribution.h"
#include "src/utils/utils.h"

VelocityDistributionBuilder::VelocityDistributionBuilder(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : DistributionMomentBuilder(simulation, diagnostics)
{
}

PetscErrorCode VelocityDistributionBuilder::build(
  const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  std::string particles;
  info.at("particles").get_to(particles);

  std::string projector;
  info.at("projector").get_to(projector);

  const auto& geometry = info.at("geometry");

  std::string name;
  geometry.at("name").get_to(name);

  Tester test;
  FieldView::Region xreg_aabb;
  xreg_aabb.dim = 2;
  xreg_aabb.dof = 1;

  if (name == "BoxGeometry") {
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
  else if (name == "CylinderGeometry") {
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
    throw std::runtime_error("Unknown geometry name " + name);
  }

  VelocityDistribution::VelocityRegion vreg{
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

  /// @todo How to name things without a logical conflicts for different geometries/parameters?
  LOG("  {} velocity distribution diagnostic is added for {}", projector, particles);

  std::string res_dir = CONFIG().out_dir + "/" + particles + "/" + projector;

  auto&& diagnostic = VelocityDistribution::create(  //
    res_dir, simulation_.get_named_particles(particles),
    projector_from_string(projector), test, xreg_aabb, vreg);

  if (!diagnostic)
    PetscFunctionReturn(PETSC_SUCCESS);

  diagnostics_.emplace_back(std::move(diagnostic));
  PetscFunctionReturn(PETSC_SUCCESS);
}
