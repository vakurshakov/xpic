#include "diagnostics_builder.h"

#include "src/implementations/basic/diagnostics/fields_energy.h"
#include "src/implementations/basic/diagnostics/field_view.h"
#include "src/utils/utils.h"

#define FIELDS_DIAGNOSTICS    (THERE_ARE_FIELDS && FIELDS_ARE_DIAGNOSED)
#define PARTICLES_DIAGNOSTICS (THERE_ARE_PARTICLES && PARTICLES_ARE_DIAGNOSED)
#define INIT_CONFIGURATION    (FIELDS_DIAGNOSTICS || PARTICLES_DIAGNOSTICS)

namespace basic {

Diagnostics_builder::Diagnostics_builder(const Simulation& simulation)
  : simulation_(simulation) {}


using Diagnostic_up = std::unique_ptr<interfaces::Diagnostic>;
using Diagnostics_vector = std::vector<Diagnostic_up>;

PetscErrorCode Diagnostics_builder::build(Diagnostics_vector& result) {
  PetscFunctionBegin;

#if INIT_CONFIGURATION
  LOG_TRACE("Building diagnostics");
  const Configuration& config = CONFIG();
  const Configuration::json_t& descriptions = config.json.at("Diagnostics");
#endif

  for (const auto& [diag_name, diag_info] : descriptions.items()) {
#if FIELDS_DIAGNOSTICS
    if (diag_name == "fields_energy") {
      LOG_INFO("Add fields energy diagnostic");
      PetscCall(build_fields_energy(diag_info, result));
    }
    else if (diag_name == "field_view") {
      LOG_INFO("Add field view diagnostic");
      PetscCall(build_fields_view(diag_info, result));
    }
#endif
  }
  result.shrink_to_fit();
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Diagnostics_builder::build_fields_energy(const Configuration::json_t& /* diag_info */, Diagnostics_vector& result) {
  PetscFunctionBegin;
  result.emplace_back(std::make_unique<Fields_energy>(CONFIG().out_dir + "/", simulation_.da_, simulation_.E_, simulation_.B_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


const Vec& Diagnostics_builder::get_field(const std::string& name) const {
  if (name == "E") return simulation_.E_;
  if (name == "B") return simulation_.B_;
  throw std::runtime_error("Unknown field name!");
}

/// @todo Lots of Field_view specific utilities, maybe should be moved into a single class
Axis get_component(const std::string& name) {
  if (name == "x") return X;
  if (name == "y") return Y;
  if (name == "z") return Z;
  throw std::runtime_error("Unknown component name!");
}

struct Field_description {
  std::string field_name;
  std::string component_name;
  Field_view::Region region;
};

using Fields_description = std::vector<Field_description>;

PetscErrorCode parse_field_info(const Configuration::json_t& json, Field_description& desc) {
  PetscFunctionBegin;
  /// @todo turn it into try-catch and print the usage info
  json.at("field").get_to(desc.field_name);
  json.at("comp").get_to(desc.component_name);

  /// @todo can be skipped to diagnose all three components
  desc.region.start[3] = get_component(desc.component_name);
  desc.region.size[3] = 1;

  /// @todo add check here for start.size() == 3 && size.size() == 3
  const Configuration::array_t& start = json.at("start");
  const Configuration::array_t& size = json.at("size");

  // Region in the configuration file is in global coordinates
  for (int i = 0; i < 3; ++i) {
    desc.region.start[i] = TO_STEP(start[i].get<PetscReal>(), Dx[i]);
    desc.region.size[i] = TO_STEP(size[i].get<PetscReal>(), Dx[i]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_field_description(const Field_description& desc) {
  PetscFunctionBegin;
  std::string message;

  bool is_field_name_correct = (desc.field_name == "E") || (desc.field_name == "B");
  message = "Unknown field name for Field_view diagnostics.";
  PetscCheck(is_field_name_correct, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, message.c_str());

  bool is_component_name_correct = (desc.component_name == "x") || (desc.component_name == "y") || (desc.component_name == "z");
  message = "Unknown component name for Field_view diagnostic of " + desc.field_name + " field.";
  PetscCheck(is_component_name_correct, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, message.c_str());

  const Field_view::Region& region = desc.region;
  bool is_region_in_bounds =
    (0 <= region.start[X] && region.start[X] < geom_nx) &&
    (0 <= region.start[Y] && region.start[Y] < geom_ny) &&
    (0 <= region.start[Z] && region.start[Z] < geom_nz) &&
    (0 <= (region.start[X] + region.size[X]) && (region.start[X] + region.size[X]) <= geom_nx) &&
    (0 <= (region.start[Y] + region.size[Y]) && (region.start[Y] + region.size[Y]) <= geom_ny) &&
    (0 <= (region.start[Z] + region.size[Z]) && (region.start[Z] + region.size[Z]) <= geom_nz);

  message = "Region is not in global boundaries for " + desc.field_name + desc.component_name + " diagnostic.";
  PetscCheck(is_region_in_bounds, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, message.c_str());

  bool are_sizes_positive = (region.size[X] > 0) && (region.size[Y] > 0) && (region.size[Z] > 0);
  message = "Sizes are negative for " + desc.field_name + desc.component_name + " diagnostic.";
  PetscCheck(are_sizes_positive, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, message.c_str());

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Attach diagnostic only to those processes, where `desc.region` lies
PetscErrorCode attach_field_description(const DM& da, Field_description&& desc, Fields_description& result) {
  PetscFunctionBegin;
  Vector3<PetscInt> start, size;
  PetscCall(DMDAGetCorners(da, R3DX(&start), R3DX(&size)));

  // check if the local region overlaps with global
  // const Field_view::Region& region = desc.region;
  // bool is_region_in_bounds =
  //   (0 <= region.start[X] && region.start[X] < geom_nx) &&
  //   (0 <= region.start[Y] && region.start[Y] < geom_ny) &&
  //   (0 <= region.start[Z] && region.start[Z] < geom_nz) &&
  //   (0 <= (region.start[X] + region.size[X]) && (region.start[X] + region.size[X]) <= geom_nx) &&
  //   (0 <= (region.start[Y] + region.size[Y]) && (region.start[Y] + region.size[Y]) <= geom_ny) &&
  //   (0 <= (region.start[Z] + region.size[Z]) && (region.start[Z] + region.size[Z]) <= geom_nz);
  //
  // MPI_Comm new_comm;
  // MPI_Comm_split(PETSC_COMM_WORLD, color, rank, &new_comm);

  // Region coordinates are used in C-order (z, y, x, dof)
  // inside of the Field_view, which is dictated by Petsc.
  Field_view::Region& region = desc.region;
  std::swap(region.start[0], region.start[2]);
  std::swap(region.size[0], region.size[2]);

  region.start[0] = std::min(region.start[0], start.z);
  region.start[1] = std::min(region.start[1], start.y);
  region.start[2] = std::min(region.start[2], start.x);

  region.size[0] = std::min(region.start[0], size.z);
  region.size[1] = std::min(region.start[1], size.y);
  region.size[2] = std::min(region.start[2], size.x);

  result.emplace_back(std::move(desc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Diagnostics_builder::build_fields_view(const Configuration::json_t& diag_info, Diagnostics_vector& result) {
  PetscFunctionBegin;
  Fields_description fields_desc;

  auto parse_info = [&](const Configuration::json_t& info) -> PetscErrorCode {
    PetscFunctionBegin;
    Field_description desc;
    PetscCall(parse_field_info(info, desc));
    PetscCall(check_field_description(desc));
    PetscCall(attach_field_description(simulation_.da_, std::move(desc), fields_desc));
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

  for (const Field_description& desc : fields_desc) {
    LOG_INFO("Field view diagnostic is added for {}{}", desc.field_name, desc.component_name);

    std::unique_ptr<Field_view>&& diag = std::make_unique<Field_view>(
      CONFIG().out_dir + "/" + desc.field_name + desc.component_name + "/",
      simulation_.da_, get_field(desc.field_name));

    PetscCall(diag->set_diagnosed_region(desc.region));

    result.emplace_back(std::move(diag));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}
