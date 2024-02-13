#include "field_view.h"

#include "src/utils/utils.h"
#include "src/vectors/vector_classes.h"

namespace fs = std::filesystem;

namespace basic {

Field_view::Field_view(const std::string& result_directory, const DM da, const Vec field)
  : interfaces::Diagnostic(result_directory), da_(da), field_(field) {}

PetscErrorCode Field_view::diagnose(timestep_t t) {
  PetscFunctionBeginUser;

  int time_width = std::to_string(geom_nt).size();
  std::stringstream ss;
  ss << std::setw(time_width) << std::setfill('0') << t;
  PetscCallMPI(file_.open(PETSC_COMM_WORLD, result_directory_, ss.str()));

  Vector3<PetscInt> start, size;
  PetscCall(DMDAGetCorners(da_, R3DX(&start), R3DX(&size)));

  PetscInt dof;
  PetscCall(DMDAGetDof(da_, &dof));

  constexpr PetscInt ndim = 4;
  PetscInt g_size[ndim] = {size.z, size.y, size.x, dof};
  PetscInt l_size[ndim] = {size.z, size.y, size.x, 1};
  PetscInt starts[ndim] = {0, 0, 0, 1};

  // file should be opened
  PetscCallMPI(file_.set_memview_subarray(ndim, g_size, l_size, starts));

  const PetscReal *arr;
  PetscCall(VecGetArrayRead(field_, &arr));

  PetscCallMPI(file_.write_floats(arr, (size.x * size.y * size.z)));
  PetscCallMPI(file_.close());

  PetscCall(VecRestoreArrayRead(field_, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}
