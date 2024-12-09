#include "sync_binary_file.h"

SyncBinaryFile::SyncBinaryFile()
{
  mode_ |= std::ios::binary;
}

SyncBinaryFile::SyncBinaryFile(const std::string& filename)
  : SyncBinaryFile()
{
  PetscCallVoid(SyncFile::open(filename));
}

PetscErrorCode SyncBinaryFile::write(PetscReal data)
{
  PetscFunctionBeginHot;
  PetscCall(write(1, &data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SyncBinaryFile::write(PetscInt size, const PetscReal* data)
{
  if (!is_synchronized())
    return PETSC_SUCCESS;

  PetscFunctionBeginHot;
  file_.write(reinterpret_cast<char*>(&data),
    static_cast<std::streamsize>(sizeof(PetscReal) * size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SyncBinaryFile::write_float(PetscReal data)
{
  PetscFunctionBeginHot;
  PetscCall(write_floats(1, &data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SyncBinaryFile::write_floats(PetscInt size, const PetscReal* data)
{
  if (!is_synchronized())
    return PETSC_SUCCESS;

  PetscFunctionBeginHot;
#if defined(PETSC_USE_REAL_SINGLE)
  PetscCall(write(size, data));
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
