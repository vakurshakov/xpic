#include "field_view.h"

#include "src/utils/geometries.h"
#include "src/utils/utils.h"


std::unique_ptr<FieldView> FieldView::create(
  const std::string& out_dir, DM da, Vec field, const Region& region)
{
  PetscFunctionBeginUser;
  MPI_Comm newcomm;
  PetscCallAbort(PETSC_COMM_WORLD, get_local_communicator(da, region, &newcomm));
  if (newcomm == MPI_COMM_NULL)
    PetscFunctionReturn(nullptr);

  auto* diagnostic = new FieldView(out_dir, da, field, newcomm);
  PetscCallAbort(PETSC_COMM_WORLD, diagnostic->set_data_views(region));
  PetscFunctionReturn(std::unique_ptr<FieldView>(diagnostic));
}

/// @returns Non-null communicator for those processes,
/// where region intersects with local boundaries of DM.
PetscErrorCode FieldView::get_local_communicator(
  DM da, const Region& region, MPI_Comm* newcomm)
{
  PetscFunctionBeginUser;
  Vector3I r_start(region.start);
  Vector3I r_size(region.size);
  Vector3I start;
  Vector3I size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  PetscMPIInt color =
    is_region_intersect_bounds(r_start, r_size, start, size) ? 1 : MPI_UNDEFINED;
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, color, rank, newcomm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

FieldView::FieldView(DM da, Vec field)
  : da_(da), field_(field)
{
}

FieldView::FieldView(
  const std::string& out_dir, DM da, Vec field, MPI_Comm newcomm)
  : interfaces::Diagnostic(out_dir), da_(da), field_(field), comm_(newcomm)
{
}

PetscErrorCode FieldView::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(file_.finalize());
  PetscCallMPI(MPI_Comm_free(&comm_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldView::set_data_views(const Region& region)
{
  PetscFunctionBeginUser;
  region_ = region;

  Vector4I l_start;
  Vector4I m_size;
  Vector4I g_start = region_.start;
  Vector4I f_size = region_.size;
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

  if (region_.dof > 1) {
    f_start[3] = 0;
    m_start[3] = g_start[3];
    l_size[3] = f_size[3];
  }

  PetscCall(file_.set_memview_subarray(region_.dim, m_size, l_size, m_start));
  PetscCall(file_.set_fileview_subarray(region_.dim, f_size, l_size, f_start));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode FieldView::diagnose(PetscInt t)
{
  if (t % diagnose_period_ != 0)
    return PETSC_SUCCESS;
  PetscFunctionBeginUser;
  PetscCall(file_.open(comm_, out_dir_ + "/" + format_time(t)));

  const PetscReal* arr;
  PetscCall(VecGetArrayRead(field_, &arr));

  PetscInt dof;
  Vector3I size;
  PetscCall(DMDAGetDof(da_, &dof));
  PetscCall(DMDAGetCorners(da_, REP3(nullptr), REP3_A(&size)));

  PetscCall(file_.write_floats((size[X] * size[Y] * size[Z] * dof), arr));
  PetscCall(file_.close());

  PetscCall(VecRestoreArrayRead(field_, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}
