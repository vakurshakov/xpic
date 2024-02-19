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
  PetscCall(DMDAGetCorners(da_, REP3_A(&start), REP3_A(&size)));

  PetscInt dof;
  PetscCall(DMDAGetDof(da_, &dof));

  PetscInt g_size[Region::ndim];
  g_size[X] = size[Z];
  g_size[Y] = size[Y];
  g_size[Z] = size[X];
  g_size[C] = dof;

  PetscInt l_size[Region::ndim];
  l_size[X] = size[Z];
  l_size[Y] = size[Y];
  l_size[Z] = size[X];
  l_size[C] = 1;

  PetscInt starts[Region::ndim];
  starts[X] = 0;
  starts[Y] = 0;
  starts[Z] = 0;
  starts[C] = 1;
  PetscCall(file_.set_memview_subarray(Region::ndim, g_size, l_size, starts));

  g_size[X] = geom_nz;
  g_size[Y] = geom_ny;
  g_size[Z] = geom_nx;
  g_size[C] = 1;

  starts[X] = start[Z];
  starts[Y] = start[Y];
  starts[Z] = start[X];
  starts[C] = 0;
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
  PetscCall(DMDAGetCorners(da_, REP3(NULL), REP3_A(&size)));

  PetscCall(file_.write_floats(arr, (size[X] * size[Y] * size[Z] * 4)));
  PetscCall(file_.close());

  PetscCall(VecRestoreArrayRead(field_, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}
