#ifndef SRC_UTILS_SYNC_BINARY_FILE
#define SRC_UTILS_SYNC_BINARY_FILE

#include "src/pch.h"

/**
 * @brief Represents synchronous (over `PETSC_COMM_WORLD`) binary file.
 * @note In most cases it should be used for scalar variables diagnostics.
 * @note Function signatures mimics MPI-files calls, so it can be used with `PetscCallMPI()`.
 */
class Sync_binary_file {
public:
  Sync_binary_file() = default;

  /// @brief Construct a new binary file and its directory path recursively.
  Sync_binary_file(const std::string& directory_path, const std::string& file_name);

  int write_floats(PetscReal* data, PetscInt size);
  int write_doubles(PetscReal* data, PetscInt size);
  int write_float(PetscReal data);
  int write_double(PetscReal data);

  int flush();
  int close();

private:
  std::ofstream file_;
};

#endif // SRC_UTILS_SYNC_BINARY_FILE
