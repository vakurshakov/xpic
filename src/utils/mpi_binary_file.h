#ifndef SRC_UTILS_MPI_BINARY_FILE
#define SRC_UTILS_MPI_BINARY_FILE

#include "src/pch.h"
#include "src/utils/utils.h"

/// @todo This should be made local to `FieldView` diagnostic
/// or remade with some default `PetscViewer` implementation.
/// Currently it is tightly coupled with our diagnostics.
class MPI_BinaryFile {
public:
  MPI_BinaryFile() = default;
  ~MPI_BinaryFile();

  /// @brief Construct a new binary file and its directory path recursively.
  MPI_BinaryFile(MPI_Comm comm, const std::string& filename);

  /// @brief Creates directories in its `directory_path` and opens a new binary file.
  PetscErrorCode open(MPI_Comm comm, const std::string& filename);
  PetscErrorCode flush();
  PetscErrorCode close();

  /// @note Should be set only once, other sets are no-op since type is already commited.
  PetscErrorCode set_memview_subarray(PetscInt ndim, const PetscInt sizes[],
    const PetscInt subsizes[], const PetscInt starts[]);
  PetscErrorCode set_fileview_subarray(PetscInt ndim, const PetscInt sizes[],
    const PetscInt subsizes[], const PetscInt starts[]);

  /// @brief Writes the array `data` of some `size` as floats.
  PetscErrorCode write_floats(PetscInt size, const PetscReal* data);

private:
  MPI_Comm comm_ = MPI_COMM_NULL;
  MPI_File file_ = MPI_FILE_NULL;

  MPI_Datatype memview_ = MPI_DATATYPE_NULL;
  MPI_Datatype fileview_ = MPI_DATATYPE_NULL;
};

#endif  // SRC_UTILS_BINARY_FILE
