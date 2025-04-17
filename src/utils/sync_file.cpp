#include "sync_file.h"

SyncFile::SyncFile(const std::string& filename)
{
  PetscCallAbort(PETSC_COMM_WORLD, open(filename));
}

PetscErrorCode SyncFile::open(const std::string& filename)
{
  if (!is_synchronized())
    return PETSC_SUCCESS;

  PetscFunctionBeginHot;
  PetscCall(close());

  std::filesystem::path path(filename);
  std::filesystem::create_directories(path.parent_path());

  file_.open(path, mode_);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SyncFile::flush()
{
  if (!is_synchronized())
    return PETSC_SUCCESS;

  PetscFunctionBeginHot;
  file_.flush();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SyncFile::close()
{
  if (!is_synchronized())
    return PETSC_SUCCESS;

  PetscFunctionBeginHot;
  if (file_.is_open()) {
    file_.flush();
    file_.close();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

bool SyncFile::is_synchronized()
{
  PetscMPIInt flag;
  PetscCallMPI(MPI_Initialized(&flag));
  if (!static_cast<bool>(flag))
    return true;

  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0)
    return true;

  return false;
}

