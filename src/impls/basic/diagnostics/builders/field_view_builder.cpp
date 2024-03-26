#include "field_view_builder.h"

namespace basic {

Field_view_builder::Field_view_builder(const Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : Diagnostic_builder(simulation, diagnostics) {}

PetscErrorCode Field_view_builder::build(const Configuration::json_t& diag_info) {
  PetscFunctionBeginUser;

  auto parse_info = [&](const Configuration::json_t& info) -> PetscErrorCode {
    PetscFunctionBeginUser;
    Field_description desc;
    PetscCall(parse_field_info(info, desc));
    PetscCall(check_field_description(desc));
    PetscCall(attach_field_description(std::move(desc)));
    PetscFunctionReturn(PETSC_SUCCESS);
  };

  if (!diag_info.is_array()) {
    PetscCall(parse_info(diag_info));
  }
  else {
    for (const Configuration::json_t& info : diag_info) {
      PetscCall(parse_info(info));
    }
  }

  for (const Field_description& desc : fields_desc_) {
    LOG_INFO("Field view diagnostic is added for {}{}", desc.field_name, desc.component_name);

    std::string res_dir = CONFIG().out_dir + "/" + desc.field_name + desc.component_name + "/";

    std::unique_ptr<Field_view>&& diag = std::make_unique<Field_view>(
      desc.comm, res_dir, simulation_.da_, get_field(desc.field_name));

    PetscCall(diag->set_diagnosed_region(desc.region));

    diagnostics_.emplace_back(std::move(diag));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Field_view_builder::parse_field_info(const Configuration::json_t& json, Field_description& desc) {
  PetscFunctionBeginUser;
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

    Vector3<PetscReal> start = parse_vector(json, "start");
    Vector3<PetscReal> size = parse_vector(json, "size");

    for (int i = 0; i < 3; ++i) {
      desc.region.start[i] = TO_STEP(start[i], Dx[i]);
      desc.region.size[i] = TO_STEP(size[i], Dx[i]);
    }
  }
  catch (const std::exception& e) {
    message = e.what();
    message += usage_message();
    throw std::runtime_error(message);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Field_view_builder::check_field_description(const Field_description& desc) {
  PetscFunctionBeginUser;
  std::string message;

  const Vector3<PetscInt>& r_start = desc.region.start;
  const Vector3<PetscInt>& r_size = desc.region.size;
  bool is_region_in_global_bounds = is_region_within_bounds(r_start, r_size, 0, Geom_n);
  message = "Region is not in global boundaries for " + desc.field_name + desc.component_name + " diagnostic.";
  PetscCheck(is_region_in_global_bounds, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, message.c_str());

  bool are_sizes_positive = (r_size[X] > 0) && (r_size[Y] > 0) && (r_size[Z] > 0);
  message = "Sizes are invalid for " + desc.field_name + desc.component_name + " diagnostic.";
  PetscCheck(are_sizes_positive, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, message.c_str());

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Attach diagnostic only to those processes, where `desc.region` lies
PetscErrorCode Field_view_builder::attach_field_description(Field_description&& desc) {
  PetscFunctionBeginUser;
  Vector3<PetscInt> start;
  Vector3<PetscInt> size;
  PetscCall(DMDAGetCorners(simulation_.da_, REP3_A(&start), REP3_A(&size)));

  const Vector3<PetscInt>& r_start = desc.region.start;
  const Vector3<PetscInt>& r_size = desc.region.size;
  bool is_local_start_in_bounds = is_region_intersect_bounds(r_start, r_size, start, size);

  PetscMPIInt color = is_local_start_in_bounds ? 1 : MPI_UNDEFINED;

  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  MPI_Comm new_comm;
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, color, rank, &new_comm));

  if (!is_local_start_in_bounds)
    PetscFunctionReturn(PETSC_SUCCESS);

  desc.comm = new_comm;

  fields_desc_.emplace_back(std::move(desc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}
