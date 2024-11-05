#include "configuration.h"

#include "src/utils/utils.h"

namespace fs = std::filesystem;

Configuration Configuration::config;

const Configuration& Configuration::get()
{
  return config;
}

void Configuration::init(const std::string& config_path)
{
  config.config_path_ = config_path;

  std::ifstream file(config_path);
  config.json = json_t::parse(file);

  const json_t& json = config.json;
  json.at("Out_dir").get_to(config.out_dir);

  const json_t& geometry = json.at("Geometry");
  geometry.at("dx").get_to(dx);
  geometry.at("dy").get_to(dy);
  geometry.at("dz").get_to(dz);
  geometry.at("dt").get_to(dt);
  Dx[0] = dx;
  Dx[1] = dy;
  Dx[2] = dz;

  geometry.at("size_x").get_to(geom_x);
  geometry.at("size_y").get_to(geom_y);
  geometry.at("size_z").get_to(geom_z);
  geometry.at("size_t").get_to(geom_t);
  Geom[0] = geom_x;
  Geom[1] = geom_y;
  Geom[2] = geom_z;

  geom_nx = TO_STEP(geom_x, dx);
  geom_ny = TO_STEP(geom_y, dy);
  geom_nz = TO_STEP(geom_z, dz);
  geom_nt = TO_STEP(geom_t, dt);
  Geom_n[0] = geom_nx;
  Geom_n[1] = geom_ny;
  Geom_n[2] = geom_nz;

  PetscReal diagnose_period_wp;
  geometry.at("diagnose_period").get_to(diagnose_period_wp);

  diagnose_period = TO_STEP(diagnose_period_wp, dt);
}


/* static */ void Configuration::save(const std::string& to)
{
  save(config.config_path_, to, fs::copy_options::overwrite_existing);
}


/* static */ void Configuration::save_sources(const std::string& to)
{
  save("src/", to,
    fs::copy_options::overwrite_existing | fs::copy_options::recursive);
}


/* static */ void Configuration::save(
  const std::string& from, const std::string& to, fs::copy_options options)
{
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank != 0)
    return;

  try {
    fs::create_directories(config.out_dir + "/" + to + "/");
    fs::copy(from, config.out_dir + "/" + to + "/", options);
  }
  catch (const fs::filesystem_error& ex) {
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


void Configuration::get_boundaries_type(
  DMBoundaryType& bx, DMBoundaryType& by, DMBoundaryType& bz)
{
  const Configuration::json_t& geometry = config.json.at("Geometry");

  std::string boundary_type_str[3];
  geometry.at("da_boundary_x").get_to(boundary_type_str[X]);
  geometry.at("da_boundary_y").get_to(boundary_type_str[Y]);
  geometry.at("da_boundary_z").get_to(boundary_type_str[Z]);

  auto to_boundary_type = [](const std::string& str) {
    if (str == "DM_BOUNDARY_PERIODIC")
      return DM_BOUNDARY_PERIODIC;
    if (str == "DM_BOUNDARY_GHOSTED")
      return DM_BOUNDARY_GHOSTED;
    return DM_BOUNDARY_NONE;
  };

  bx = to_boundary_type(boundary_type_str[X]);
  by = to_boundary_type(boundary_type_str[Y]);
  bz = to_boundary_type(boundary_type_str[Z]);
}


void Configuration::get_processors(PetscInt& px, PetscInt& py, PetscInt& pz)
{
  const Configuration::json_t& geometry = config.json.at("Geometry");
  geometry.at("da_processors_x").get_to(px);
  geometry.at("da_processors_y").get_to(py);
  geometry.at("da_processors_z").get_to(pz);
}

