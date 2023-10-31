#include "binary_file.h"

namespace fs = std::filesystem;


Binary_file::Binary_file(const std::string& directory_path, const std::string& file_name) {
  fs::create_directories(directory_path);
  file_.open(directory_path + "/" + file_name + ".bin",
    std::ios::out | std::ios::trunc | std::ios::binary);
}

/* static */ Binary_file Binary_file::from_timestep(const std::string& directory_path, timestep_t timestep) {
  int time_width = std::to_string(TIME).size();

  std::stringstream ss;
  ss << std::setw(time_width) << std::setfill('0') << timestep;

  Binary_file result(directory_path, ss.str());

  return result;
}

/* static */ Binary_file Binary_file::from_backup(
    const std::string& directory_path, const std::string& file_name, int byte_offset) {
  fs::create_directories(directory_path);

  Binary_file result;

  result.file_.open(directory_path + "/" + file_name + ".bin",
    std::ios::in | std::ios::out | std::ios::binary);

  result.file_.seekp(-byte_offset, std::ios::end);

  return result;
}

void Binary_file::write_as_floats(double* data, size_t size) {
  std::vector<float> tmp;
  tmp.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    tmp.emplace_back(static_cast<float>(data[i]));
  }
  file_.write(reinterpret_cast<char*>(tmp.data()), sizeof(float) * size);
}

void Binary_file::write_as_doubles(double* data, size_t size) {
  file_.write(reinterpret_cast<char*>(&data), sizeof(double) * size);
}

void Binary_file::flush() {
  file_.flush();
}

void Binary_file::close() {
  flush();
  file_.close();
}

