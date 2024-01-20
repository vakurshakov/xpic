#include "configuration.h"

namespace fs = std::filesystem;

Configuration Configuration::instance_;

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
