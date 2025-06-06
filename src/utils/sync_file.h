#ifndef SRC_UTILS_SYNC_FILE_H
#define SRC_UTILS_SYNC_FILE_H

#include "src/pch.h"
#include "src/utils/utils.h"

/// @todo Rewrite the class as a wrapper around `PetscViewer`
class SyncFile {
public:
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

#endif  // SRC_UTILS_SYNC_FILE_H
