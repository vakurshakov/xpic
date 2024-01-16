#include "configuration.h"

namespace fs = std::filesystem;

Configuration::Configuration(const std::string& config_path)
{
  config_path_ = config_path;

  std::ifstream file(config_path);
  item_ = json::parse(file);

  out_dir = get("Out_dir");

  dx = get<double>("Geometry.dx");
  dy = get<double>("Geometry.dy");
  dz = get<double>("Geometry.dz");
  dt = get<double>("Geometry.dt");

  size_lx = get<double>("Geometry.size_x");
  size_ly = get<double>("Geometry.size_y");
  size_lz = get<double>("Geometry.size_y");

#define TO_STEP(dim, ds) static_cast<int>(round(dim / ds))
  size_nx = TO_STEP(size_lx, dx);
  size_ny = TO_STEP(size_ly, dy);
  size_nz = TO_STEP(size_lz, dy);

  time = TO_STEP(get<double>("Geometry.time"), dt);
  diagnose_period = TO_STEP(get<double>("Geometry.diagnose_period"), dt);
#undef TO_STEP
}

void Configuration::save(const std::string& to) const {
  save(config_path_, to, fs::copy_options::overwrite_existing);
}

void Configuration::save_sources(const std::string& to) const {
  save("src/", to, fs::copy_options::overwrite_existing | fs::copy_options::recursive);
}

void Configuration::save(const std::string& from, const std::string& to, fs::copy_options options) const {
  try {
    fs::create_directories(out_dir + "/" + to + "/");
    fs::copy(from, out_dir + "/" + to + "/", options);
  }
  catch(const fs::filesystem_error& ex) {
    std::stringstream ss;

    ss << "what():  " << ex.what() << '\n'
       << "path1(): " << ex.path1() << '\n'
       << "path2(): " << ex.path2() << '\n'
       << "code().value():    " << ex.code().value() << '\n'
       << "code().message():  " << ex.code().message() << '\n'
       << "code().category(): " << ex.code().category().name() << '\n';

    throw std::runtime_error(ss.str());
  }
}
