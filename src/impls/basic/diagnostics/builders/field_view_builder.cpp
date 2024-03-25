#include "field_view_builder.h"

namespace basic {

Field_view_builder::Field_view_builder(const Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : Diagnostic_builder(simulation, diagnostics) {}

PetscErrorCode Field_view_builder::build(const Configuration::json_t& diag_info) {
  PetscFunctionBegin;

  auto parse_info = [&](const Configuration::json_t& info) -> PetscErrorCode {
    PetscFunctionBegin;
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
  PetscFunctionBegin;
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
  PetscFunctionBegin;
  std::string message;

  bool is_field_name_correct =
    desc.field_name == "E" ||
    desc.field_name == "B" ||
    desc.field_name == "J";
  message = "Unknown field name for Field_view diagnostics.";
  PetscCheck(is_field_name_correct, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, message.c_str());

  bool is_component_name_correct =
    desc.component_name == "x" ||
    desc.component_name == "y" ||
    desc.component_name == "z" ||
    desc.component_name == "";
  message = "Unknown component name for Field_view diagnostic of " + desc.field_name + " field.";
  PetscCheck(is_component_name_correct, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, message.c_str());

  const Field_view::Region& reg = desc.region;
  bool is_region_in_global_bounds =
    (0 <= reg.start[X] && reg.start[X] < geom_nx) &&
    (0 <= reg.start[Y] && reg.start[Y] < geom_ny) &&
    (0 <= reg.start[Z] && reg.start[Z] < geom_nz) &&
    (0 <= (reg.start[X] + reg.size[X]) && (reg.start[X] + reg.size[X]) <= geom_nx) &&
    (0 <= (reg.start[Y] + reg.size[Y]) && (reg.start[Y] + reg.size[Y]) <= geom_ny) &&
    (0 <= (reg.start[Z] + reg.size[Z]) && (reg.start[Z] + reg.size[Z]) <= geom_nz);

  message = "Region is not in global boundaries for " + desc.field_name + desc.component_name + " diagnostic.";
  PetscCheck(is_region_in_global_bounds, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, message.c_str());

  bool are_sizes_positive = (reg.size[X] > 0) && (reg.size[Y] > 0) && (reg.size[Z] > 0);
  message = "Sizes are invalid for " + desc.field_name + desc.component_name + " diagnostic.";
  PetscCheck(are_sizes_positive, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, message.c_str());

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Attach diagnostic only to those processes, where `desc.region` lies
PetscErrorCode Field_view_builder::attach_field_description(Field_description&& desc) {
  PetscFunctionBegin;
  Vector3<PetscInt> start;
  Vector3<PetscInt> end;
  PetscCall(DMDAGetCorners(simulation_.da_, REP3_A(&start), REP3_A(&end)));
  end += start;

  Vector3<PetscInt> r_start = desc.region.start;
  Vector3<PetscInt> r_end = desc.region.size;
  r_end += r_start;

  // checking intersection between local domain and `desc.region`
  bool is_local_start_in_bounds =
    r_start[X] < end[X] && r_end[X] > start[X] &&
    r_start[Y] < end[Y] && r_end[Y] > start[Y] &&
    r_start[Z] < end[Z] && r_end[Z] > start[Z];

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
