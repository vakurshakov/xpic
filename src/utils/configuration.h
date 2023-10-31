#ifndef SRC_UTILS_CONFIGURATION_HPP
#define SRC_UTILS_CONFIGURATION_HPP

#include <nlohmann/json.hpp>

#include "src/pch.h"

class Config_item {
 public:
  template<typename T = std::string>
  T get(const std::string& key = "") const;

  template<typename T = std::string>
  T get(const std::string& key, T default_value) const;

  Config_item get_item(const std::string& key) const;

  bool contains(const std::string& key) const;
  bool is_array(const std::string& key = "") const;

  using item_parser = std::function<void(const Config_item&)>;
  void for_each(const std::string& key, item_parser func) const;
  void for_each(item_parser func) const;

 protected:
  using json = nlohmann::ordered_json;
  json item_;

  json::json_pointer to_pointer(const std::string& key) const;

  Config_item() = default;
  Config_item(const json& item);

  Config_item(const Config_item&) = delete;
  Config_item& operator=(const Config_item&) = delete;

  Config_item(Config_item&&) = delete;
  Config_item& operator=(Config_item&&) = delete;
};


class Configuration : public Config_item {
 public:
  static const Configuration& instance(const char* config_path = "");
  static const std::string& out_dir();

  void save(const std::string& to = "") const;
  void save_sources(const std::string& to = "") const;
  void init_geometry() const;

 private:
  std::string out_dir_;
  std::string config_path_;

  void save(const std::string& from, const std::string& to,
    std::filesystem::copy_options options) const;

  Configuration(const char* config_path);

  Configuration(const Configuration&) = delete;
  Configuration& operator=(const Configuration&) = delete;

  Configuration(Configuration&&) = delete;
  Configuration& operator=(Configuration&&) = delete;
};

#include "configuration.inl"

#endif // SRC_UTILS_CONFIGURATION_HPP
