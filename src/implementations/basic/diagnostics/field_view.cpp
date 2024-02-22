#include "field_view.h"

#include "src/utils/utils.h"
#include "src/vectors/vector_classes.h"

namespace fs = std::filesystem;

namespace basic {

Field_view::Field_view(MPI_Comm comm, const std::string& result_directory, const DM& da, const Vec& field)
  : interfaces::Diagnostic(result_directory), da_(da), field_(field), comm_(comm) {}

PetscErrorCode Field_view::set_diagnosed_region(const Region& region) {
  PetscFunctionBegin;
  Vector4<PetscInt> l_start, g_start = region.start;
  Vector4<PetscInt> m_size, f_size = region.size;
  PetscCall(DMDAGetCorners(da_, REP3_A(&l_start), REP3_A(&m_size)));
  PetscCall(DMDAGetDof(da_, &m_size[3]));

  l_start.to_petsc_order();
  g_start.to_petsc_order();
  m_size.to_petsc_order();
  f_size.to_petsc_order();

  Vector4<PetscInt> m_start;
  m_start[0] = std::max(g_start[0], l_start[0]);
  m_start[1] = std::max(g_start[1], l_start[1]);
  m_start[2] = std::max(g_start[2], l_start[2]);

  Vector4<PetscInt> l_size;
  l_size[0] = std::min(g_start[0] + f_size[0], l_start[0] + m_size[0]) - m_start[0];
  l_size[1] = std::min(g_start[1] + f_size[1], l_start[1] + m_size[1]) - m_start[1];
  l_size[2] = std::min(g_start[2] + f_size[2], l_start[2] + m_size[2]) - m_start[2];
  l_size[3] = f_size[3];

  Vector4<PetscInt> f_start = m_start;
  f_start -= g_start;  // file start is in global coordinates, but we remove offset
  f_start[3] = 0;  // if one component is written

  m_start -= l_start;  // memory start is in local coordinates
  m_start[3] = g_start[3];

  PetscCall(file_.set_memview_subarray(Region::ndim, m_size, l_size, m_start));
  PetscCall(file_.set_fileview_subarray(Region::ndim, f_size, l_size, f_start));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Field_view::diagnose(timestep_t t) {
  PetscFunctionBeginUser;

  int time_width = std::to_string(geom_nt).size();
  std::stringstream ss;
  ss << std::setw(time_width) << std::setfill('0') << t;
  PetscCall(file_.open(comm_, result_directory_, ss.str()));

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
