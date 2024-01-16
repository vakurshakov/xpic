#include "configuration.h"

/* static */ inline const Configuration& Configuration::instance(const std::string& config_path) {
  static Configuration instance(config_path);
  return instance;
}

inline nlohmann::json::json_pointer Configuration::to_pointer(const std::string& key) const {
  std::string copy(key);
  std::replace(copy.begin(), copy.end(), '.', '/');
  return json::json_pointer{"/" + copy};
}

template<typename T>
inline T Configuration::get(const std::string& key) const {
  return item_.at(to_pointer(key)).get<T>();
}

template<typename T>
inline T Configuration::get(const std::string& key, T default_value) const {
  auto pointer = to_pointer(key);
  if (item_.contains(pointer)) {
    return item_.at(pointer).get<T>();
  }
  return default_value;
}
