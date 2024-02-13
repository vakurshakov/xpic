#ifndef SRC_UTILS_MPI_BINARY_FILE
#define SRC_UTILS_MPI_BINARY_FILE

#include "src/pch.h"

/**
 * @brief
 * @note Function signatures mimics MPI-files calls, so it can be used with `PetscCallMPI()`.
 */
class MPI_binary_file {
public:
  MPI_binary_file() = default;
  ~MPI_binary_file();

  /// @brief Construct a new binary file and its directory path recursively.
  MPI_binary_file(MPI_Comm comm, const std::string& directory_path, const std::string& file_name);

  /// @brief Creates directories in its `directory_path` and opens a new binary file.
  int open(MPI_Comm comm, const std::string& directory_path, const std::string& file_name);
  int flush();
  int close();

  /// should be set only once, other set(s) are no-op since type is already commited
  /// PetscReal type (with MPIU_REAL size) is the only assumed type
  int set_memview_subarray(int ndim, const int sizes[], const int subsizes[], const int starts[]);

  int write_floats(const PetscReal* data, PetscInt size);

private:
  MPI_Comm comm_ = MPI_COMM_NULL;
  MPI_File file_ = MPI_FILE_NULL;

  MPI_Datatype memview_ = MPI_DATATYPE_NULL;
  MPI_Datatype fileview_ = MPI_DATATYPE_NULL;
};

#endif // SRC_UTILS_BINARY_FILE
