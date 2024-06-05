#include "field_view.h"

#include "src/utils/utils.h"
#include "src/vectors/vector4.h"

namespace basic {

Field_view::Field_view(MPI_Comm comm, const std::string& out_dir, const DM& da, const Vec& field)
  : interfaces::Diagnostic(out_dir), da_(da), field_(field), comm_(comm) {}

PetscErrorCode Field_view::set_diagnosed_region(const Region& region) {
  PetscFunctionBeginUser;
  Vector4I l_start, g_start = region.start;
  Vector4I m_size, f_size = region.size;
  PetscCall(DMDAGetCorners(da_, REP3_A(&l_start), REP3_A(&m_size)));
  PetscCall(DMDAGetDof(da_, &m_size[3]));

  l_start.swap_order();
  g_start.swap_order();
  m_size.swap_order();
  f_size.swap_order();

  Vector4I m_start = max(g_start, l_start);
  Vector4I l_size = min(g_start + f_size, l_start + m_size) - m_start;
  Vector4I f_start = m_start;

  f_start -= g_start;  // file start is in global coordinates, but we remove offset
  m_start -= l_start;  // memory start is in local coordinates

  f_start[3] = 0;
  m_start[3] = g_start[3];
  l_size[3] = f_size[3];

  PetscCall(file_.set_memview_subarray(Region::ndim, m_size, l_size, m_start));
  PetscCall(file_.set_fileview_subarray(Region::ndim, f_size, l_size, f_start));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Field_view::diagnose(timestep_t t) {
  if (t % diagnose_period != 0)
    PetscFunctionReturn(PETSC_SUCCESS);
  PetscFunctionBeginUser;

  int time_width = std::to_string(geom_nt).size();
  std::stringstream ss;
  ss << std::setw(time_width) << std::setfill('0') << t;
  PetscCall(file_.open(comm_, out_dir_, ss.str()));

  const PetscReal *arr;
  PetscCall(VecGetArrayRead(field_, &arr));

  Vector3I size;
  PetscCall(DMDAGetCorners(da_, REP3(nullptr), REP3_A(&size)));

  PetscCall(file_.write_floats(arr, (size[X] * size[Y] * size[Z] * Region::ndim)));
  PetscCall(file_.close());

  PetscCall(VecRestoreArrayRead(field_, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}
