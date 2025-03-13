#include "field_view_builder.h"

#include "src/utils/configuration.h"
#include "src/utils/vector_utils.h"

FieldViewBuilder::FieldViewBuilder(const interfaces::Simulation& simulation,
  std::vector<Diagnostic_up>& diagnostics)
  : DiagnosticBuilder(simulation, diagnostics)
{
}

PetscErrorCode FieldViewBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  FieldView::Region region;
  region.dim = 4;
  region.dof = 3;

  std::string field;
  info.at("field").get_to(field);

  std::string comp;
  region.start[3] = 0;
  region.size[3] = 3;

  if (info.contains("comp")) {
    info.at("comp").get_to(comp);
    region.start[3] = get_component(comp);
    region.size[3] = 1;
  }

  const Configuration::json_t& region_info = info.at("region");
  parse_region_start_size(region_info, region, field + comp);

  std::string suffix;
  parse_res_dir_suffix(region_info, suffix);

  LOG("  Field view diagnostic is added for {}, suffix: {}", field + comp, suffix);

  std::string res_dir = CONFIG().out_dir + "/" + field + comp + suffix + "/";

  auto&& diagnostic = FieldView::create(
    res_dir, simulation_.world.da, simulation_.get_named_vector(field), region);

  if (!diagnostic)
    PetscFunctionReturn(PETSC_SUCCESS);

  diagnostics_.emplace_back(std::move(diagnostic));
  PetscFunctionReturn(PETSC_SUCCESS);
}

void FieldViewBuilder::parse_region_start_size(const Configuration::json_t& info,
  FieldView::Region& region, const std::string& name)
{
  Vector3R start{0.0};
  Vector3R size{Geom};

  std::string type = "3D";

  if (info.contains("type"))
    info.at("type").get_to(type);

  PetscInt dim = (type == "3D") ? 3 : (type == "2D") ? 2 : -1;

  if (dim < 0)
    throw std::runtime_error("Incorrect type is used for " + name + " .");

  if (info.contains("start"))
    start = parse_vector(info, "start", dim);

  if (info.contains("size"))
    size = parse_vector(info, "size", dim);

  if (type == "2D") {
    std::string plane;
    info.at("plane").get_to(plane);

    if (plane == "X") {
      start = Vector3R{info.at("position"), start[X], start[Y]};
      size = Vector3R{dx, size[X], size[Y]};
    }
    else if (plane == "Y") {
      start = Vector3R{start[X], info.at("position"), start[Y]};
      size = Vector3R{size[X], dy, size[Y]};
    }
    else if (plane == "Z") {
      start = Vector3R{start[X], start[Y], info.at("position")};
      size = Vector3R{size[X], size[Y], dz};
    }
  }

  for (PetscInt i = 0; i < 3; ++i) {
    region.start[i] = ROUND_STEP(start[i], Dx[i]);
    region.size[i] = ROUND_STEP(size[i], Dx[i]);
  }

  check_region(vector_cast(region.start), vector_cast(region.size), name);
}

void FieldViewBuilder::parse_res_dir_suffix(
  const Configuration::json_t& info, std::string& suffix)
{
  std::string type = "3D";

  if (info.contains("type"))
    info.at("type").get_to(type);

  if (type == "2D") {
    std::string plane;
    info.at("plane").get_to(plane);

    PetscReal position;
    info.at("position").get_to(position);

    suffix += "_Plane" + plane;

    auto parse = [&](Axis dir) {
      // Analogous to `interfaces::Diagnostics::format_time()`
      auto width = (PetscInt)std::to_string((PetscInt)Geom[dir]).size();
      std::stringstream ss;
      ss.width(width);
      ss.fill('0');
      ss << std::to_string(ROUND_STEP(position, Dx[dir]));
      suffix += "_" + ss.str() + "cwpe";
    };

    if (plane == "X")
      parse(X);
    else if (plane == "Y")
      parse(Y);
    else if (plane == "Z")
      parse(Z);
  }
}
