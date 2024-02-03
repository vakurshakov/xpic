#ifndef SRC_UTILS_CONFIGURATION_HPP
#define SRC_UTILS_CONFIGURATION_HPP

#include <nlohmann/json.hpp>

#include "src/pch.h"

class Configuration {
public:
  inline static void init(const std::string& config_path);
  inline static const Configuration& get();

  void save(const std::string& to = "") const;
  void save_sources(const std::string& to = "") const;

  std::string out_dir;

  using json = nlohmann::ordered_json;

  template<typename T = std::string>
  T get(const std::string& key = "") const;

  template<typename T = std::string>
  T get(const std::string& key, T default_value) const;

private:
  std::string config_path_;

  json item_;
  json::json_pointer to_pointer(const std::string& key) const;

  void save(const std::string& from, const std::string& to, std::filesystem::copy_options options) const;

  Configuration() = default;
  static Configuration instance_;
};

#include "configuration.inl"

#define CONFIG()  ::Configuration::get()

#endif // SRC_UTILS_CONFIGURATION_HPP
