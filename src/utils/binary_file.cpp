#include "binary_file.h"

namespace fs = std::filesystem;


Binary_file::Binary_file(const std::string& directory_path, const std::string& file_name) {
  fs::create_directories(directory_path);
  file_.open(directory_path + "/" + file_name + ".bin", std::ios::out | std::ios::trunc | std::ios::binary);
}

/* static */ Binary_file Binary_file::from_timestep(const std::string& directory_path, timestep_t timestep) {
  int time_width = std::to_string(geom_nt).size();

  std::stringstream ss;
  ss << std::setw(time_width) << std::setfill('0') << timestep;

  Binary_file result(directory_path, ss.str());

  return result;
}

template<typename T>
void write_data(std::ofstream& os, PetscReal* data, PetscInt size) {
  std::vector<T> tmp(size);
  for (PetscInt i = 0; i < size; ++i) {
    tmp[i] = static_cast<T>(data[i]);
  }
  os.write(reinterpret_cast<char*>(tmp.data()), sizeof(T) * tmp.size());
}

void Binary_file::write_as_floats(PetscReal* data, PetscInt size) {
#if defined(PETSC_USE_REAL_SINGLE)
  file_.write(reinterpret_cast<char*>(&data), sizeof(float) * size);
#else
  write_data<float>(file_, data, size);
#endif
}

void Binary_file::write_as_doubles(PetscReal* data, PetscInt size) {
#if defined(PETSC_USE_REAL_DOUBLE)
  file_.write(reinterpret_cast<char*>(&data), sizeof(double) * size);
#else
  write_data<double>(file_, data, size);
#endif
}

void Binary_file::write_float(PetscReal data) {
  float float_data = static_cast<float>(data);
  file_.write(reinterpret_cast<char*>(&float_data), sizeof(float));
}

void Binary_file::write_double(PetscReal data) {
  double double_data = static_cast<double>(data);
  file_.write(reinterpret_cast<char*>(&double_data), sizeof(double));
}

void Binary_file::flush() {
  file_.flush();
}

void Binary_file::close() {
  flush();
  file_.close();
}

