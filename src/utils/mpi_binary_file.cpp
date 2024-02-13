#include "mpi_binary_file.h"

namespace fs = std::filesystem;

MPI_binary_file::MPI_binary_file(MPI_Comm comm, const std::string& directory_path, const std::string& file_name) {
  open(comm, directory_path, file_name);
}

MPI_binary_file::~MPI_binary_file() {
  if (memview_ != MPI_DATATYPE_NULL)
    MPI_Type_free(&memview_);

  if (fileview_ != MPI_DATATYPE_NULL)
    MPI_Type_free(&fileview_);

  if (file_ != MPI_FILE_NULL)
    close();

  comm_ = MPI_COMM_NULL;
}

int MPI_binary_file::open(MPI_Comm comm, const std::string& directory_path, const std::string& file_name) {
  PetscFunctionBeginHot;
  comm_ = comm;

  std::string filename = directory_path + "/" + file_name + ".bin";

  int rank;
  PetscCallMPI(MPI_Comm_rank(comm_, &rank));
  if (rank == 0) {
    fs::create_directories(directory_path);
    fs::remove(filename);
  }
  PetscCallMPI(MPI_Barrier(comm_));

  PetscCallMPI(MPI_File_open(comm_, filename.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file_));
  PetscFunctionReturn(MPI_SUCCESS);
}

int MPI_binary_file::flush() {
  PetscFunctionBeginHot;
  PetscCallMPI(MPI_File_sync(file_));
  PetscFunctionReturn(MPI_SUCCESS);
}

int MPI_binary_file::close() {
  PetscFunctionBeginHot;
  PetscCallMPI(MPI_File_close(&file_));
  PetscFunctionReturn(MPI_SUCCESS);
}


int MPI_binary_file::set_memview_subarray(int ndim, const int sizes[], const int subsizes[], const int starts[]) {
  PetscFunctionBegin;
  PetscCallMPI(MPI_Type_create_subarray(ndim, sizes, subsizes, starts, MPI_ORDER_C, MPIU_REAL, &memview_));
  PetscCallMPI(MPI_Type_commit(&memview_));

  int rank;
  PetscCallMPI(MPI_Comm_rank(comm_, &rank));

  MPI_Offset disp = rank * sizeof(MPIU_REAL);
  for (int i = 0; i < ndim; ++i) { disp *= subsizes[i]; }
  PetscCallMPI(MPI_File_set_view(file_, disp, MPIU_REAL, MPIU_REAL, "native", MPI_INFO_NULL));
  PetscFunctionReturn(MPI_SUCCESS);
}

int MPI_binary_file::write_floats(const PetscReal* data, PetscInt /* size */) {
  PetscFunctionBeginHot;
#if 1 // defined(PETSC_USE_REAL_SINGLE)
  PetscCallMPI(MPI_File_write_all(file_, data, 1, memview_, MPI_STATUS_IGNORE));
#else
#endif
  PetscFunctionReturn(MPI_SUCCESS);
}
