#include "sync_binary_file.h"

namespace fs = std::filesystem;

Sync_binary_file::Sync_binary_file(const std::string& directory_path, const std::string& file_name) {
  open(directory_path, file_name);
}

#undef SYNC_GUARD
#define SYNC_GUARD                                      \
  int rank;                                             \
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank)); \
  if (rank != 0) return MPI_SUCCESS;                    \

int Sync_binary_file::open(const std::string& directory_path, const std::string& file_name) {
  SYNC_GUARD;
  PetscFunctionBeginHot;
  PetscCallMPI(close());
  fs::create_directories(directory_path);
  file_.open(directory_path + "/" + file_name + ".bin", std::ios::out | std::ios::trunc | std::ios::binary);
  PetscFunctionReturn(MPI_SUCCESS);
}

int Sync_binary_file::flush() {
  SYNC_GUARD;
  PetscFunctionBeginHot;
  file_.flush();
  PetscFunctionReturn(MPI_SUCCESS);
}

int Sync_binary_file::close() {
  SYNC_GUARD;
  PetscFunctionBeginHot;
  if (file_.is_open()) {
    file_.flush();
    file_.close();
  }
  PetscFunctionReturn(MPI_SUCCESS);
}


int Sync_binary_file::write_floats(const PetscReal* data, PetscInt size) {
  SYNC_GUARD;
  PetscFunctionBeginHot;
#if defined(PETSC_USE_REAL_SINGLE)
  file_.write(reinterpret_cast<char*>(&data), sizeof(float) * size);
#else
  std::vector<float> buf(size);
  for (PetscInt i = 0; i < size; ++i) { buf[i] = static_cast<float>(data[i]); }
  file_.write(reinterpret_cast<char*>(buf.data()), sizeof(float) * buf.size());
#endif
  PetscFunctionReturn(MPI_SUCCESS);
}

int Sync_binary_file::write_float(PetscReal data) {
  SYNC_GUARD;
  PetscFunctionBeginHot;
  float float_data = static_cast<float>(data);
  file_.write(reinterpret_cast<char*>(&float_data), sizeof(float));
  PetscFunctionReturn(MPI_SUCCESS);
}
