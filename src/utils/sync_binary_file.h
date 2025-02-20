#ifndef SRC_UTILS_SYNC_BINARY_FILE_H
#define SRC_UTILS_SYNC_BINARY_FILE_H

#include "src/utils/sync_file.h"

class SyncBinaryFile : public SyncFile {
public:
  SyncBinaryFile();
  SyncBinaryFile(const std::string& filename);

  PetscErrorCode write(PetscReal data);
  PetscErrorCode write(PetscInt size, const PetscReal* data);

  PetscErrorCode write_float(PetscReal data);
  PetscErrorCode write_floats(PetscInt size, const PetscReal* data);
};

#endif  // SRC_UTILS_SYNC_BINARY_FILE_H
