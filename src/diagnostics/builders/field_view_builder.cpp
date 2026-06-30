#include "field_view_builder.h"

#include "src/utils/configuration.h"
#include "src/utils/vector_utils.h"

FieldViewBuilder::FieldViewBuilder(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : DiagnosticBuilder(simulation, diagnostics)
{
}

PetscErrorCode FieldViewBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  std::string field;
  info.at("field").get_to(field);

  std::string out_dir = field;

  DM da;
  Vec f;
  Region region;
  region.dim = 4;
  region.start = Vector4I(0);
  region.size = Vector4I(geom_nx, geom_ny, geom_nz, -1);

  parse_field(info, da, f, region, field);

  if (info.contains("region"))
    parse_region(info.at("region"), region, field);

  if (info.contains("out_dir"))
    info.at("out_dir").get_to(out_dir);

  LOG("  field view diagnostic is added for {}, output directory: {}", field, out_dir);

  if (auto&& diagnostic =
        FieldView::create(CONFIG().out_dir + "/" + out_dir, da, f, region)) {
    diagnostics_.emplace_back(std::move(diagnostic));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

void FieldViewBuilder::parse_field(const Configuration::json_t& info, DM& da,
  Vec& f, Region& region, const std::string& name)
{
  DM da_field = simulation_.da;
  DM da_rho = simulation_.da_rho;
  DM da_mflux = simulation_.da_mflux;

  std::map<std::string, std::pair<DM, Vec>> map{
    {"E", {da_field, simulation_.E}},
    {"B", {da_field, simulation_.B}},
    {"J", {da_field, simulation_.J}},
    {"B0", {da_field, simulation_.B0}},
  };

  for (auto&& sort : simulation_.particles_) {
    std::string J_name = sort->parameters.sort_name + "/J";
    std::string rho_name = sort->parameters.sort_name + "/rho";
    std::string mflux_name = sort->parameters.sort_name + "/mflux";
    map[J_name] = std::make_pair(da_field, sort->J);
    map[rho_name] = std::make_pair(da_rho, sort->rho[1]);
    map[mflux_name] = std::make_pair(da_mflux, sort->mflux);
  }

  if (!map.contains(name))
    throw std::runtime_error("Incorrect name is used for " + name + ".");

  std::tie(da, f) = map.at(name);

  PetscInt dof;
  DMDAGetDof(da, &dof);

  region.dof = dof;
  region.size[3] = dof;
}

void FieldViewBuilder::parse_region(
  const Configuration::json_t& info, Region& region, const std::string& name)
{
  Vector3R start{0.0};
  Vector3R size{Geom};

  std::string type = "3D";

  if (auto it = info.find("type"); it != info.end())
    it->get_to(type);

  PetscInt dim = (type == "3D") ? 3 : (type == "2D") ? 2 : -1;

  if (dim < 0)
    throw std::runtime_error("Incorrect type is used for " + name + ".");

  if (info.contains("start"))
    start = parse_vector(info, "start");

  if (info.contains("size"))
    size = parse_vector(info, "size");

  if (type == "2D") {
    std::string plane;
    PetscReal position;
    parse_plane_position(info, plane, position);

    if (plane == "X") {
      start = Vector3R{position, start[Y], start[Z]};
      size = Vector3R{dx, size[Y], size[Z]};
    }
    else if (plane == "Y") {
      start = Vector3R{start[X], position, start[Z]};
      size = Vector3R{size[X], dy, size[Z]};
    }
    else if (plane == "Z") {
      start = Vector3R{start[X], start[Y], position};
      size = Vector3R{size[X], size[Y], dz};
    }
  }

  for (PetscInt i = 0; i < 3; ++i) {
    region.start[i] = FLOOR_STEP(start[i], Dx[i]);
    region.size[i] = FLOOR_STEP(size[i], Dx[i]);
  }

  check_region(region, name);
}

void FieldViewBuilder::parse_plane_position(
  const Configuration::json_t& info, std::string& plane, PetscReal& position)
{
  info.at("plane").get_to(plane);

  if (plane == "X")
    position = 0.5 * geom_x;
  else if (plane == "Y")
    position = 0.5 * geom_y;
  else if (plane == "Z")
    position = 0.5 * geom_z;

  if (info.contains("position"))
    info.at("position").get_to(position);
}

void FieldViewBuilder::check_region(
  const Region& region, const std::string& name) const
{
  Vector3I start = vector_cast(region.start);
  Vector3I size = vector_cast(region.size);

  if (bool success = is_region_within_bounds(start, size, 0, Geom_n); !success) {
    throw std::runtime_error(
      "Region is not in global boundaries for " + name + " diagnostic.");
  }

  if (bool success = (size[X] > 0) && (size[Y] > 0) && (size[Z] > 0); !success)
    throw std::runtime_error("Sizes are invalid for " + name + " diagnostic.");
}
