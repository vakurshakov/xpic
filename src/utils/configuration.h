#ifndef SRC_UTILS_CONFIGURATION_HPP
#define SRC_UTILS_CONFIGURATION_HPP

#include <filesystem>

#include <petscdmtypes.h>  // for DMBoundaryType
#include <petscsystypes.h>  // for PetscInt

#include <nlohmann/json.hpp>

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
   * @returns Read only reference to `Configuration`.
   */
  static const Configuration& get();

  /**
   * @brief Overwrites underlying `json` and `out_dir`. Should be used before any manipulations.
   * @warning This is mostly a utility to set up tests, we recommended to avoid it's direct usage.
   */
  static void overwrite(Configuration::json_t&& json);

  /**
   * @brief Initializes configuration class instance. Should be used before any manipulations.
   * @param config_path Location of json file, provided by argv[1].
   */
  static void init(const std::string& config_path);

  /// @brief Stores the configuration json file. Overwrites the existing one, if present.
  static void save(const std::string& out_dir);

  /// @brief Stores the entire `src/` directory. Overwrites the existing one, if present.
  static void save_sources(const std::string& out_dir);

  static bool is_loaded_from_backup();

  static void get_boundaries_type(
    DMBoundaryType& bx, DMBoundaryType& by, DMBoundaryType& bz);

  static void get_processors(  //
    PetscInt& px, PetscInt& py, PetscInt& pz);

private:
  std::string config_path_;

  static void save(const std::string& from, const std::string& our_dir,
    std::filesystem::copy_options options);

  Configuration() = default;
  static Configuration config;
};

#define CONFIG() ::Configuration::get()

#endif  // SRC_UTILS_CONFIGURATION_HPP
