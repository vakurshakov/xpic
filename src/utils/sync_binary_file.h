#ifndef SRC_UTILS_SYNC_BINARY_FILE
#define SRC_UTILS_SYNC_BINARY_FILE

#include "src/pch.h"

/**
 * @brief Represents synchronous (over `PETSC_COMM_WORLD`) binary file.
 * @note In most cases it should be used for scalar variables diagnostics.
 */
class Sync_binary_file {
public:
  Sync_binary_file() = default;

  /// @brief Creates directories in its `directory_path` and opens a new binary file.
  Sync_binary_file(const std::string& directory_path, const std::string& file_name);

  /// @brief Creates directories in its `directory_path` and opens a new binary file.
  int open(const std::string& directory_path, const std::string& file_name);
  int flush();
  int close();

  int write_floats(const PetscReal* data, PetscInt size);
  int write_float(PetscReal data);

private:
  std::ofstream file_;
};

#endif // SRC_UTILS_SYNC_BINARY_FILE
