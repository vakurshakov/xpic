#include "configuration.h"

#include "src/utils/utils.h"

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
  json.at("OutputDirectory").get_to(config.out_dir);

  const json_t& geometry = json.at("Geometry");
  geometry.at("dx").get_to(dx);
  geometry.at("dy").get_to(dy);
  geometry.at("dz").get_to(dz);
  geometry.at("dt").get_to(dt);
  Dx[0] = dx;
  Dx[1] = dy;
  Dx[2] = dz;

  /// @todo Read with dimension?
  geometry.at("x").get_to(geom_nx);
  geometry.at("y").get_to(geom_ny);
  geometry.at("z").get_to(geom_nz);
  geometry.at("t").get_to(geom_nt);
  Geom_n[0] = geom_nx;
  Geom_n[1] = geom_ny;
  Geom_n[2] = geom_nz;

  geom_x = geom_nx * dx;
  geom_y = geom_ny * dy;
  geom_z = geom_nz * dz;
  geom_t = geom_nt * dt;
  Geom[0] = geom_x;
  Geom[1] = geom_y;
  Geom[2] = geom_z;

  geometry.at("diagnose_period").get_to(diagnose_period);
}


/* static */ void Configuration::save()
{
  save(config.config_path_, std::filesystem::copy_options::overwrite_existing);
}


/* static */ void Configuration::save_sources()
{
  save("src/",
    std::filesystem::copy_options::overwrite_existing |
      std::filesystem::copy_options::recursive);
}


/* static */ void Configuration::save(
  const std::string& from, std::filesystem::copy_options options)
{
  PetscInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank != 0)
    return;

  try {
    std::filesystem::create_directories(config.out_dir + "/");
    std::filesystem::copy(from, config.out_dir + "/", options);
  }
  catch (const std::filesystem::filesystem_error& ex) {
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

