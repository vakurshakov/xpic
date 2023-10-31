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

  /**
   * @brief Construct a new binary file based on the backup.
   *
   * @param directory_path Where the diagnostic was stored.
   * @param file_name Name of the diagnostics you are trying to load.
   * @param byte_offset The offset made from the last position after opening.
   *
   * @warning This method should be used only to restore single file diagnostics.
   */
  static Binary_file from_backup(const std::string& directory_path, const std::string& file_name, int byte_offset);

  void write_as_floats(double* data, size_t size);
  void write_as_doubles(double* data, size_t size);

  void flush();
  void close();

 private:
  std::ofstream file_;
};

#endif // SRC_UTILS_BINARY_FILE
