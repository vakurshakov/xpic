#include "mpi_binary_file.h"

MPI_BinaryFile::MPI_BinaryFile(MPI_Comm comm, const std::string& directory_path,
  const std::string& file_name)
{
  PetscCallVoid(open(comm, directory_path, file_name));
}

MPI_BinaryFile::~MPI_BinaryFile()
{
  if (memview_ != MPI_DATATYPE_NULL)
    MPI_Type_free(&memview_);

  if (fileview_ != MPI_DATATYPE_NULL)
    MPI_Type_free(&fileview_);

  if (file_ != MPI_FILE_NULL)
    close();

  if (comm_ != MPI_COMM_NULL)
    MPI_Comm_free(&comm_);
}

PetscErrorCode MPI_BinaryFile::open(MPI_Comm comm,
  const std::string& directory_path, const std::string& file_name)
{
  PetscFunctionBeginHot;
  comm_ = comm;

  std::string filename = directory_path + "/" + file_name + ".bin";

  PetscInt rank;
  PetscCallMPI(MPI_Comm_rank(comm_, &rank));
  if (rank == 0) {
    std::filesystem::create_directories(directory_path);
    std::filesystem::remove(filename);
  }
  PetscCallMPI(MPI_Barrier(comm_));

  PetscCallMPI(MPI_File_open(comm_, filename.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file_));
  PetscCallMPI(MPI_File_set_view(file_, 0, MPI_FLOAT, fileview_, "native", MPI_INFO_NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MPI_BinaryFile::flush()
{
  PetscFunctionBeginHot;
  PetscCallMPI(MPI_File_sync(file_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MPI_BinaryFile::close()
{
  PetscFunctionBeginHot;
  PetscCallMPI(MPI_File_close(&file_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode create_subarray(PetscInt ndim, const PetscInt sizes[],
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

PetscErrorCode MPI_BinaryFile::set_memview_subarray(PetscInt ndim,
  const PetscInt sizes[], const PetscInt subsizes[], const PetscInt starts[])
{
  PetscFunctionBeginUser;
  PetscCall(create_subarray(ndim, sizes, subsizes, starts, &memview_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MPI_BinaryFile::set_fileview_subarray(PetscInt ndim,
  const PetscInt sizes[], const PetscInt subsizes[], const PetscInt starts[])
{
  PetscFunctionBeginUser;
  PetscCall(create_subarray(ndim, sizes, subsizes, starts, &fileview_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MPI_BinaryFile::write_floats(PetscInt size, const PetscReal* data)
{
  PetscFunctionBeginHot;
#if defined(PETSC_USE_REAL_SINGLE)
  PetscCallMPI(MPI_File_write_all(file_, data, 1, memview_, MPI_STATUS_IGNORE));
#else
  /// @todo It works, but it can be expensive for small datasets. We should exploit memview_.
  std::vector<float> buf(size);
  for (PetscInt i = 0; i < size; ++i)
    buf[i] = static_cast<float>(data[i]);
  PetscCallMPI(MPI_File_write_all(file_, buf.data(), 1, memview_, MPI_STATUS_IGNORE));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
