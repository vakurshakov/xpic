#include "configuration.h"

/* static */ inline void Configuration::init(const std::string& config_path)
{
  auto& config = instance_;

  config.config_path_ = config_path;

  std::ifstream file(config_path);
  config.item_ = json::parse(file);

  config.out_dir = config.get("Out_dir");

  config.dx = config.get<double>("Geometry.dx");
  config.dy = config.get<double>("Geometry.dy");
  config.dz = config.get<double>("Geometry.dz");
  config.dt = config.get<double>("Geometry.dt");

  config.size_lx = config.get<double>("Geometry.size_x");
  config.size_ly = config.get<double>("Geometry.size_y");
  config.size_lz = config.get<double>("Geometry.size_y");

#define TO_STEP(dim, ds) static_cast<int>(round(dim / ds))
  config.size_nx = TO_STEP(config.size_lx, config.dx);
  config.size_ny = TO_STEP(config.size_ly, config.dy);
  config.size_nz = TO_STEP(config.size_lz, config.dy);

  config.time = TO_STEP(config.get<double>("Geometry.time"), config.dt);
  config.diagnose_period = TO_STEP(config.get<double>("Geometry.diagnose_period"), config.dt);
#undef TO_STEP
}

/* static */ inline const Configuration& Configuration::get() {
  return instance_;
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
