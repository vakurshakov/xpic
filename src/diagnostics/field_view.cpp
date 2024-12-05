#include "field_view.h"

#include "src/utils/region_operations.h"
#include "src/utils/utils.h"


std::unique_ptr<FieldView> FieldView::create(
  const std::string& out_dir, DM da, Vec field, const Region& region)
{
  PetscFunctionBeginUser;
  MPI_Comm newcomm;
  PetscCallThrow(get_local_communicator(da, region, &newcomm));
  if (newcomm == MPI_COMM_NULL)
    PetscFunctionReturn(nullptr);

  auto* diagnostic = new FieldView(out_dir, da, field, newcomm);
  PetscCallThrow(diagnostic->set_data_views(region));
  PetscFunctionReturn(std::unique_ptr<FieldView>(diagnostic));
}


/// @returns Non-null communicator for those processes,
/// where region intersects with local boundaries of DM.
PetscErrorCode FieldView::get_local_communicator(
  DM da, const Region& region, MPI_Comm* newcomm)
{
  PetscFunctionBeginUser;
  Vector3I r_start(region.start), r_size(region.size), start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  PetscMPIInt color =
    is_region_intersect_bounds(r_start, r_size, start, size) ? 1 : MPI_UNDEFINED;
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, color, rank, newcomm));
  PetscFunctionReturn(PETSC_SUCCESS);
}


FieldView::FieldView(
  const std::string& out_dir, DM da, Vec field, MPI_Comm newcomm)
  : interfaces::Diagnostic(out_dir), da_(da), field_(field), comm_(newcomm)
{
}


PetscErrorCode FieldView::set_data_views(const Region& region)
{
  PetscFunctionBeginUser;
  region_ = region;

  Vector4I l_start, g_start = region_.start;
  Vector4I m_size, f_size = region_.size;
  PetscCall(DMDAGetCorners(da_, REP3_A(&l_start), REP3_A(&m_size)));
  PetscCall(DMDAGetDof(da_, &m_size[3]));

  l_start.swap_order();
  g_start.swap_order();
  m_size.swap_order();
  f_size.swap_order();

  Vector4I m_start = max(g_start, l_start);
  Vector4I l_size = min(g_start + f_size, l_start + m_size) - m_start;
  Vector4I f_start = m_start;

  // file start is in global coordinates, but we remove offset
  f_start -= g_start;

  // memory start is in local coordinates
  m_start -= l_start;

  if (region_.dim > 3) {
    f_start[3] = 0;
    m_start[3] = g_start[3];
    l_size[3] = f_size[3];
  }

  PetscCall(file_.set_memview_subarray(region_.dim, m_size, l_size, m_start));
  PetscCall(file_.set_fileview_subarray(region_.dim, f_size, l_size, f_start));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode FieldView::diagnose(timestep_t t)
{
  if (t % diagnose_period != 0)
    PetscFunctionReturn(PETSC_SUCCESS);
  PetscFunctionBeginUser;

  auto time_width = static_cast<PetscInt>(std::to_string(geom_nt).size());
  std::stringstream ss;
  ss << std::setw(time_width) << std::setfill('0') << t;
  PetscCall(file_.open(comm_, out_dir_, ss.str()));

  const PetscReal* arr;
  PetscCall(VecGetArrayRead(field_, &arr));

  Vector3I size;
  PetscCall(DMDAGetCorners(da_, REP3(nullptr), REP3_A(&size)));

  PetscCall(file_.write_floats((size[X] * size[Y] * size[Z] * region_.dof), arr));
  PetscCall(file_.close());

  PetscCall(VecRestoreArrayRead(field_, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}
