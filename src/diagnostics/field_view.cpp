#include "field_view.h"

#include "src/utils/geometries.h"
#include "src/utils/utils.h"


std::unique_ptr<FieldView> FieldView::create(
  const std::string& out_dir, DM da, Vec field, const Region& reg)
{
  PetscFunctionBeginUser;
  MPI_Comm newcomm;
  PetscCallAbort(PETSC_COMM_WORLD, World::create_local_comm(da, reg, &newcomm));

  if (newcomm == MPI_COMM_NULL)
    PetscFunctionReturn(nullptr);

  auto* diagnostic = new FieldView(out_dir, da, field, newcomm);
  PetscCallAbort(PETSC_COMM_WORLD, diagnostic->set_data_views(reg));
  PetscFunctionReturn(std::unique_ptr<FieldView>(diagnostic));
}

FieldView::FieldView(DM da, Vec field)
  : da(da), field(field)
{
}

FieldView::FieldView(
  const std::string& out_dir, DM da, Vec field, MPI_Comm newcomm)
  : interfaces::Diagnostic(out_dir), da(da), field(field), comm(newcomm)
{
}

PetscErrorCode FieldView::finalize()
{
  PetscFunctionBeginUser;
  if (memview != MPI_DATATYPE_NULL)
    PetscCallMPI(MPI_Type_free(&memview));

  if (fileview != MPI_DATATYPE_NULL)
    PetscCallMPI(MPI_Type_free(&fileview));

  if (file != MPI_FILE_NULL)
    PetscCall(close());

  if (comm != MPI_COMM_NULL)
    PetscCallMPI(MPI_Comm_free(&comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldView::diagnose(PetscInt t)
{
  if (t % diagnose_period_ != 0)
    return PETSC_SUCCESS;
  PetscFunctionBeginUser;
  PetscCall(open(out_dir_ + "/" + format_time(t)));

  const PetscReal* arr;
  PetscCall(VecGetArrayRead(field, &arr));

  PetscInt dof;
  Vector3I size;
  PetscCall(DMDAGetDof(da, &dof));
  PetscCall(DMDAGetCorners(da, REP3(nullptr), REP3_A(&size)));

  PetscCall(write(size[X] * size[Y] * size[Z] * dof, arr));
  PetscCall(close());

  PetscCall(VecRestoreArrayRead(field, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode FieldView::set_data_views(const Region& reg)
{
  PetscFunctionBeginUser;
  region = reg;

  Vector4I l_start;
  Vector4I m_size;
  Vector4I g_start = region.start;
  Vector4I f_size = region.size;
  PetscCall(DMDAGetCorners(da, REP3_A(&l_start), REP3_A(&m_size)));
  PetscCall(DMDAGetDof(da, &m_size[3]));

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

  if (region.dof > 1) {
    f_start[3] = 0;
    m_start[3] = g_start[3];
    l_size[3] = f_size[3];
  }

  PetscCall(create_subarray(region.dim, m_size, l_size, m_start, &memview));
  PetscCall(create_subarray(region.dim, f_size, l_size, f_start, &fileview));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldView::create_subarray(PetscInt ndim, const PetscInt sizes[],
  const PetscInt subsizes[], const PetscInt starts[], MPI_Datatype* type)
{
  PetscFunctionBeginUser;
  PetscMPIInt d;
  PetscCall(PetscMPIIntCast(ndim, &d));

  std::vector<PetscMPIInt> sz(d);
  std::vector<PetscMPIInt> st(d);
  std::vector<PetscMPIInt> sb(d);
  for (PetscMPIInt i = 0; i < d; ++i) {
    PetscCall(PetscMPIIntCast(sizes[i], &sz[i]));
    PetscCall(PetscMPIIntCast(starts[i], &st[i]));
    PetscCall(PetscMPIIntCast(subsizes[i], &sb[i]));
  }
  PetscCallMPI(MPI_Type_create_subarray(d, sz.data(), sb.data(), st.data(), MPI_ORDER_C, MPI_FLOAT, type));
  PetscCallMPI(MPI_Type_commit(type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldView::open(const std::string& filename)
{
  PetscFunctionBeginHot;
  std::filesystem::path fname(filename);

  PetscInt rank;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    std::filesystem::create_directories(fname.parent_path());
    std::filesystem::remove(filename);
  }
  PetscCallMPI(MPI_Barrier(comm));
  PetscCallMPI(MPI_File_open(comm, fname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file));
  PetscCallMPI(MPI_File_set_view(file, 0, MPI_FLOAT, fileview, "native", MPI_INFO_NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldView::flush()
{
  PetscFunctionBeginHot;
  PetscCallMPI(MPI_File_sync(file));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldView::close()
{
  PetscFunctionBeginHot;
  PetscCallMPI(MPI_File_close(&file));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldView::write(PetscInt size, const PetscReal* data)
{
  PetscFunctionBeginHot;
#if defined(PETSC_USE_REAL_SINGLE)
  PetscCall(write_floats(data));
#else
  /// @todo It works, but it can be expensive for small datasets. We should exploit memview_.
  std::vector<float> buf(size);
  for (PetscInt i = 0; i < size; ++i)
    buf[i] = static_cast<float>(data[i]);
  PetscCall(write_floats(buf.data()));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldView::write_floats(const float* data)
{
  PetscFunctionBeginHot;
  PetscCallMPI(MPI_File_write_all(file, data, 1, memview, MPI_STATUS_IGNORE));
  PetscFunctionReturn(PETSC_SUCCESS);
}
