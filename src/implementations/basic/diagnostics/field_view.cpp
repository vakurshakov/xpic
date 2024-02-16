#include "field_view.h"

#include "src/utils/utils.h"
#include "src/vectors/vector_classes.h"

namespace fs = std::filesystem;

namespace basic {

Field_view::Field_view(const std::string& result_directory, const DM& da, const Vec& field)
  : interfaces::Diagnostic(result_directory), da_(da), field_(field) {}

PetscErrorCode Field_view::set_diagnosed_region(const Region& /* region */) {
  PetscFunctionBegin;

  Vector3<PetscInt> start, size;
  PetscCall(DMDAGetCorners(da_, R3DX(&start), R3DX(&size)));

  PetscInt dof;
  PetscCall(DMDAGetDof(da_, &dof));

  PetscInt g_size[Region::ndim];
  g_size[0] = size.z;
  g_size[1] = size.y;
  g_size[2] = size.x;
  g_size[3] = dof;

  PetscInt l_size[Region::ndim];
  l_size[0] = size.z;
  l_size[1] = size.y;
  l_size[2] = size.x;
  l_size[3] = 1;

  PetscInt starts[Region::ndim];
  starts[0] = 0;
  starts[1] = 0;
  starts[2] = 0;
  starts[3] = 1;
  PetscCall(file_.set_memview_subarray(Region::ndim, g_size, l_size, starts));

  g_size[0] = geom_nz;
  g_size[1] = geom_ny;
  g_size[2] = geom_nx;
  g_size[3] = 1;

  starts[0] = start.z;
  starts[1] = start.y;
  starts[2] = start.x;
  starts[3] = 0;
  PetscCall(file_.set_fileview_subarray(Region::ndim, g_size, l_size, starts));
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
  PetscCall(DMDAGetCorners(da_, R3C(NULL), R3DX(&size)));

  PetscCall(file_.write_floats(arr, (size.x * size.y * size.z * 4)));
  PetscCall(file_.close());

  PetscCall(VecRestoreArrayRead(field_, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}
