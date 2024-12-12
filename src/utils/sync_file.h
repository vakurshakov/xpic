#ifndef SRC_UTILS_SYNC_FILE
#define SRC_UTILS_SYNC_FILE

#include "src/pch.h"
#include "src/utils/utils.h"

class SyncFile {
public:
  /// @warning std::ofstream is not copyable.
  DEFAULT_MOVABLE(SyncFile);

  SyncFile() = default;
  ~SyncFile() = default;

  /// @brief Creates directories in `parent_path()` of the `filename` and opens a new file.
  SyncFile(const std::string& filename);

  /// @brief Creates directories in `parent_path()` of the `filename` and opens a new file.
  PetscErrorCode open(const std::string& filename);
  PetscErrorCode flush();
  PetscErrorCode close();

  std::ofstream& operator()()
  {
    return file_;
  }

protected:
  static bool is_synchronized();

  std::ofstream::openmode mode_ = std::ios::out | std::ios::trunc;
  std::ofstream file_;
};

#endif  // SRC_UTILS_SYNC_BINARY_FILE
