#ifndef SRC_UTILS_BINARY_FILE
#define SRC_UTILS_BINARY_FILE

#include "src/pch.h"

class Binary_file {
 public:
  Binary_file() = default;

  /// @brief Construct a new binary file and its directory path recursively.
  Binary_file(const std::string& directory_path, const std::string& file_name);

  /// @brief Construct a new binary file with the name based on timestep.
  static Binary_file from_timestep(const std::string& directory_path, timestep_t timestep);

  void write_as_floats(PetscReal* data, PetscInt size);
  void write_as_doubles(PetscReal* data, PetscInt size);

  void write_float(PetscReal data);
  void write_double(PetscReal data);

  void flush();
  void close();

 private:
  std::ofstream file_;
};

#endif // SRC_UTILS_BINARY_FILE
