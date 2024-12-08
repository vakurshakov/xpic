#ifndef SRC_UTILS_SYNC_BINARY_FILE
#define SRC_UTILS_SYNC_BINARY_FILE

#include "src/pch.h"
#include "src/utils/utils.h"

/**
 * @note In most cases it should be used for scalar variables diagnostics.
 * @todo Should be inherited from MPI_binary_file to reuse it's logic.
 */
class SyncBinaryFile {
public:
  /// @warning std::ofstream is not copyable.
  DEFAULT_MOVABLE(SyncBinaryFile);

  SyncBinaryFile() = default;
  ~SyncBinaryFile() = default;

  /// @brief Creates directories in its `directory_path` and opens a new binary file.
  SyncBinaryFile(const std::string& directory_path, const std::string& file_name);

  /// @brief Creates directories in its `directory_path` and opens a new binary file.
  PetscErrorCode open(
    const std::string& directory_path, const std::string& file_name);
  PetscErrorCode flush();
  PetscErrorCode close();

  PetscErrorCode write_floats(PetscInt size, const PetscReal* data);

private:
  std::ofstream file_;
};

#endif  // SRC_UTILS_SYNC_BINARY_FILE
