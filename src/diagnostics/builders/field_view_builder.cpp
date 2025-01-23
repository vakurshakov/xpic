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

  Vector3R start{0.0};
  Vector3R size{Geom};

  if (info.contains("start"))
    start = parse_vector(info, "start");

  if (info.contains("size"))
    size = parse_vector(info, "size");

  for (PetscInt i = 0; i < 3; ++i) {
    region.start[i] = TO_STEP(start[i], Dx[i]);
    region.size[i] = TO_STEP(size[i], Dx[i]);
  }

  check_region(vector_cast(region.start), vector_cast(region.size), field + comp);

  LOG("Field view diagnostic is added for {}", field + comp);

  std::string res_dir = CONFIG().out_dir + "/" + field + comp + "/";

  auto&& diagnostic = FieldView::create(res_dir, simulation_.world_.da,
    simulation_.get_named_vector(field), region);

  if (!diagnostic)
    PetscFunctionReturn(PETSC_SUCCESS);

  diagnostics_.emplace_back(std::move(diagnostic));
  PetscFunctionReturn(PETSC_SUCCESS);
}
