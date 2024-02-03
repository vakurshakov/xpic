#include "configuration.h"

/* static */ inline void Configuration::init(const std::string& config_path)
{
  auto& config = instance_;

  config.config_path_ = config_path;

  std::ifstream file(config_path);
  config.item_ = json::parse(file);

  config.out_dir = config.get("Out_dir");

  dx = config.get<double>("Geometry.dx");
  dy = config.get<double>("Geometry.dy");
  dz = config.get<double>("Geometry.dz");
  dt = config.get<double>("Geometry.dt");

  size_lx = config.get<double>("Geometry.size_x");
  size_ly = config.get<double>("Geometry.size_y");
  size_lz = config.get<double>("Geometry.size_z");
  size_lt = config.get<double>("Geometry.size_t");

#define TO_STEP(dim, ds) static_cast<int>(round(dim / ds))
  size_nx = TO_STEP(size_lx, dx);
  size_ny = TO_STEP(size_ly, dy);
  size_nz = TO_STEP(size_lz, dz);
  size_nt = TO_STEP(size_lt, dt);

  diagnose_period = TO_STEP(config.get<double>("Geometry.diagnose_period"), dt);
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
