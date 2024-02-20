#include "field_view.h"

#include "src/utils/utils.h"
#include "src/vectors/vector_classes.h"

namespace fs = std::filesystem;

namespace basic {

Field_view::Field_view(MPI_Comm comm, const std::string& result_directory, const DM& da, const Vec& field)
  : interfaces::Diagnostic(result_directory), da_(da), field_(field), comm_(comm) {}

PetscErrorCode Field_view::set_diagnosed_region(const Region& region) {
  PetscFunctionBegin;
  Vector4<PetscInt> start, size;
  PetscCall(DMDAGetCorners(da_, REP3_A(&start), REP3_A(&size)));

  start.to_petsc_order();
  start[3] = 0; // if one component is written

  size.to_petsc_order();
  PetscCall(DMDAGetDof(da_, &size[3]));

  Vector4<PetscInt> l_size;
  l_size[0] = size[0]; // if we are writing 3D array
  l_size[1] = size[1]; // same
  l_size[2] = size[2]; // same
  l_size[3] = region.size[3];

  Vector4<PetscInt> f_size = region.size;
  f_size.to_petsc_order();

  PetscCall(file_.set_memview_subarray(Region::ndim, size, l_size, region.start));
  PetscCall(file_.set_fileview_subarray(Region::ndim, f_size, l_size, start));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Field_view::diagnose(timestep_t t) {
  PetscFunctionBeginUser;

  int time_width = std::to_string(geom_nt).size();
  std::stringstream ss;
  ss << std::setw(time_width) << std::setfill('0') << t;
  PetscCall(file_.open(PETSC_COMM_WORLD, result_directory_, ss.str()));

  const PetscReal *arr;
  PetscCall(VecGetArrayRead(field_, &arr));

  Vector3<PetscInt> size;
  PetscCall(DMDAGetCorners(da_, REP3(nullptr), REP3_A(&size)));

  PetscCall(file_.write_floats(arr, (size[X] * size[Y] * size[Z] * Region::ndim)));
  PetscCall(file_.close());

  PetscCall(VecRestoreArrayRead(field_, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}
