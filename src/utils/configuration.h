#ifndef SRC_UTILS_CONFIGURATION_HPP
#define SRC_UTILS_CONFIGURATION_HPP

#include <nlohmann/json.hpp>

#include <petscdm.h>  // for DMBoundaryType

#include "src/pch.h"

class Configuration {
public:
  /// @note Using ordered json in case of order dependent configuration.
  using json_t = nlohmann::ordered_json;
  using array_t = nlohmann::ordered_json::array_t;

  /// @brief The main storage for the configuration inputted by `config.json` file.
  json_t json;

  /// @brief Root directory of the simulation output.
  std::string out_dir;

  /**
   * @brief Retrieves the only instance (process local) of the Configuration class.
   * @return Read only reference to `Configuration`.
   */
  static const Configuration& get();

  /**
   * @brief Initializes configuration class instance.
   * Should be used before any get/save operations.
   * @param config_path Location of json file, provided by argv[1].
   */
  static void init(const std::string& config_path);

  /**
   * @brief Stores the configuration json file.
   * Overwrites the existing one, if present.
   * @param to Location relative to `out_dir` where json file would be stored.
   */
  static void save(const std::string& to = "");

  /**
   * @brief Stores the entire `src/` directory.
   * @param to Location of resulting src directory relative to `out_dir`.
   */
  static void save_sources(const std::string& to = "");

  /// @brief Getter of boundaries type, reads the config at
  /// Geometry.da_boundary_{x, y, z}.
  static void get_boundaries_type(
    DMBoundaryType& bx, DMBoundaryType& by, DMBoundaryType& bz);

  /// @brief Getter of processors number, reads the config at
  /// Geometry.da_processors_{x, y, z}.
  static void get_processors(PetscInt& px, PetscInt& py, PetscInt& pz);

private:
  std::string config_path_;

  static void save(const std::string& from, const std::string& to,
    std::filesystem::copy_options options);

  Configuration() = default;
  static Configuration config;
};

#define CONFIG() ::Configuration::get()

#endif  // SRC_UTILS_CONFIGURATION_HPP
