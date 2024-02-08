#include "sync_binary_file.h"

namespace fs = std::filesystem;

#define SYNC_GUARD                        \
  int rank;                               \
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank); \
  if (rank != 0) return                   \

Sync_binary_file::Sync_binary_file(const std::string& directory_path, const std::string& file_name) {
  SYNC_GUARD;
  fs::create_directories(directory_path);
  file_.open(directory_path + "/" + file_name + ".bin", std::ios::out | std::ios::trunc | std::ios::binary);
}

#undef SYNC_GUARD
#define SYNC_GUARD                                      \
  int rank;                                             \
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank)); \
  if (rank != 0) return MPI_SUCCESS;                    \

template<typename T>
int write_data(std::ofstream& os, PetscReal* data, PetscInt size) {
  SYNC_GUARD;
  PetscFunctionBeginHot;
  std::vector<T> tmp(size);
  for (PetscInt i = 0; i < size; ++i) {
    tmp[i] = static_cast<T>(data[i]);
  }
  os.write(reinterpret_cast<char*>(tmp.data()), sizeof(T) * tmp.size());
  PetscFunctionReturn(MPI_SUCCESS);
}

int Sync_binary_file::write_floats(PetscReal* data, PetscInt size) {
  SYNC_GUARD;
  PetscFunctionBeginHot;
#if defined(PETSC_USE_REAL_SINGLE)
  file_.write(reinterpret_cast<char*>(&data), sizeof(float) * size);
  PetscFunctionReturn(MPI_SUCCESS);
#else
  PetscCallMPI(write_data<float>(file_, data, size));
  PetscFunctionReturn(MPI_SUCCESS);
#endif
}

int Sync_binary_file::write_doubles(PetscReal* data, PetscInt size) {
  SYNC_GUARD;
  PetscFunctionBeginHot;
#if defined(PETSC_USE_REAL_DOUBLE)
  file_.write(reinterpret_cast<char*>(&data), sizeof(double) * size);
  PetscFunctionReturn(MPI_SUCCESS);
#else
  PetscCallMPI(write_data<double>(file_, data, size));
  PetscFunctionReturn(MPI_SUCCESS);
#endif
}

int Sync_binary_file::write_float(PetscReal data) {
  SYNC_GUARD;
  PetscFunctionBeginHot;
  float float_data = static_cast<float>(data);
  file_.write(reinterpret_cast<char*>(&float_data), sizeof(float));
  PetscFunctionReturn(MPI_SUCCESS);
}

int Sync_binary_file::write_double(PetscReal data) {
  SYNC_GUARD;
  PetscFunctionBeginHot;
  double double_data = static_cast<double>(data);
  file_.write(reinterpret_cast<char*>(&double_data), sizeof(double));
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
  PetscCallMPI(flush());
  file_.close();
  PetscFunctionReturn(MPI_SUCCESS);
}

