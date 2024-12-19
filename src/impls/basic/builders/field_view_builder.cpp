#include "field_view_builder.h"

#include "src/utils/configuration.h"
#include "src/utils/vector_utils.h"

FieldViewBuilder::FieldViewBuilder(const interfaces::Simulation& simulation,
  std::vector<Diagnostic_up>& diagnostics)
  : DiagnosticBuilder(simulation, diagnostics)
{
}

PetscErrorCode FieldViewBuilder::build(const Configuration::json_t& diag_info)
{
  PetscFunctionBeginUser;

  auto parse_info = [&](const Configuration::json_t& info) -> PetscErrorCode {
    PetscFunctionBeginUser;
    FieldDescription desc;
    PetscCall(parse_field_info(info, desc));
    fields_desc_.emplace_back(std::move(desc));
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  if (!diag_info.is_array())
    PetscCall(parse_info(diag_info));
  else
    for (const Configuration::json_t& info : diag_info)
      PetscCall(parse_info(info));

  for (const FieldDescription& desc : fields_desc_) {
    LOG("Field view diagnostic is added for {}{}", desc.field_name, desc.component_name);

    std::string res_dir =
      CONFIG().out_dir + "/" + desc.field_name + desc.component_name + "/";

    if (auto&& diag = FieldView::create(res_dir, simulation_.world_.da,
          get_global_vector(desc.field_name), desc.region)) {
      diagnostics_.emplace_back(std::move(diag));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldViewBuilder::parse_field_info(
  const Configuration::json_t& json, FieldDescription& desc)
{
  PetscFunctionBeginUser;
  desc.region.dim = 4;
  desc.region.dof = 3;

  std::string message;
  try {
    json.at("field").get_to(desc.field_name);

    desc.region.start[3] = 0;
    desc.region.size[3] = 3;

    if (json.contains("comp")) {
      json.at("comp").get_to(desc.component_name);
      if (!desc.component_name.empty()) {
        desc.region.start[3] = get_component(desc.component_name);
        desc.region.size[3] = 1;
      }
    }

    Vector3R start = parse_vector(json, "start");
    Vector3R size = parse_vector(json, "size");

    for (PetscInt i = 0; i < 3; ++i) {
      desc.region.start[i] = TO_STEP(start[i], Dx[i]);
      desc.region.size[i] = TO_STEP(size[i], Dx[i]);
    }

    PetscCall(check_region(vector_cast(desc.region.start), vector_cast(desc.region.size), desc.field_name + desc.component_name));
  }
  catch (const std::exception& e) {
    message = e.what();
    message += usage_message();
    throw std::runtime_error(message);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
