#ifndef SRC_UTILS_CONFIGURATION_HPP
#define SRC_UTILS_CONFIGURATION_HPP

#include <nlohmann/json.hpp>

#include "src/pch.h"

class Configuration {
public:
  static const Configuration& instance(const std::string& config_path = "");

  void save(const std::string& to = "") const;
  void save_sources(const std::string& to = "") const;

  std::string out_dir;

  double dx = 0.0;  // c / w_pe
  double dy = 0.0;  // c / w_pe
  double dz = 0.0;  // c / w_pe

  int size_nx = 0;  // units of dx
  int size_ny = 0;  // units of dy
  int size_nz = 0;  // units of dz

  double size_lx = 0.0;  // c / w_pe
  double size_ly = 0.0;  // c / w_pe
  double size_lz = 0.0;  // c / w_pe

  double dt = 0.0;                 // 1 / w_pe
  timestep_t time = 0;             // units of dt
  timestep_t diagnose_period = 0;  // units of dt

  using json = nlohmann::ordered_json;

  template<typename T = std::string>
  T get(const std::string& key = "") const;

  template<typename T = std::string>
  T get(const std::string& key, T default_value) const;

private:
  std::string config_path_;

  json item_;
  json::json_pointer to_pointer(const std::string& key) const;

  void save(const std::string& from, const std::string& to,
    std::filesystem::copy_options options) const;

  Configuration(const std::string& config_path);

  Configuration(const Configuration&) = delete;
  Configuration& operator=(const Configuration&) = delete;
};

#include "configuration.inl"

#endif // SRC_UTILS_CONFIGURATION_HPP
