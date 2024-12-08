#include "sync_binary_file.h"

namespace fs = std::filesystem;

SyncBinaryFile::SyncBinaryFile(
  const std::string& directory_path, const std::string& file_name)
{
  PetscCallVoid(open(directory_path, file_name));
}

#define SYNC_GUARD                                     \
  PetscMPIInt rank;                                    \
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank)); \
  if (rank != 0)                                       \
  return PETSC_SUCCESS

PetscErrorCode SyncBinaryFile::open(
  const std::string& directory_path, const std::string& file_name)
{
  SYNC_GUARD;
  PetscFunctionBeginHot;
  PetscCall(close());
  fs::create_directories(directory_path);
  file_.open(directory_path + "/" + file_name + ".bin",
    std::ios::out | std::ios::trunc | std::ios::binary);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SyncBinaryFile::flush()
{
  SYNC_GUARD;
  PetscFunctionBeginHot;
  file_.flush();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SyncBinaryFile::close()
{
  SYNC_GUARD;
  PetscFunctionBeginHot;
  if (file_.is_open()) {
    file_.flush();
    file_.close();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode SyncBinaryFile::write_floats(PetscInt size, const PetscReal* data)
{
  SYNC_GUARD;
  PetscFunctionBeginHot;
#if defined(PETSC_USE_REAL_SINGLE)
  file_.write(reinterpret_cast<char*>(&data),
    static_cast<std::streamsize>(sizeof(float) * size));
#else
  std::vector<float> buf(size);
  for (PetscInt i = 0; i < size; ++i)
    buf[i] = static_cast<float>(data[i]);
  file_.write(reinterpret_cast<char*>(buf.data()),
    static_cast<std::streamsize>(sizeof(float) * buf.size()));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef SYNC_GUARD
