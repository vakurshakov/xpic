#include "configuration.h"

#include "src/utils/utils.h"
#include "src/utils/world.h"

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

  World::set_geometry( //
    geometry.at("x").get<PetscReal>(), //
    geometry.at("y").get<PetscReal>(), //
    geometry.at("z").get<PetscReal>(), //
    geometry.at("t").get<PetscReal>(), //
    geometry.at("dx").get<PetscReal>(), //
    geometry.at("dy").get<PetscReal>(), //
    geometry.at("dz").get<PetscReal>(), //
    geometry.at("dt").get<PetscReal>(), //
    geometry.at("diagnose_period").get<PetscReal>());
}


/* static */ void Configuration::save(const std::string& out_dir)
{
  save(config.config_path_, out_dir,
    std::filesystem::copy_options::overwrite_existing);
}


/* static */ void Configuration::save_sources(const std::string& out_dir)
{
  std::filesystem::path src_dir = __FILE__;
  src_dir = src_dir.parent_path();
  src_dir = src_dir.parent_path();

  save(src_dir, out_dir,
    std::filesystem::copy_options::overwrite_existing |
      std::filesystem::copy_options::recursive);
}


/* static */ void Configuration::save(const std::string& from,
  const std::string& to, std::filesystem::copy_options options)
{
  PetscInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (rank != 0)
    return;

  try {
    std::filesystem::create_directories(to);
    std::filesystem::copy(from, to, options);
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

bool Configuration::is_loaded_from_backup()
{
  const json_t& json = config.json;

  auto&& it = json.find("SimulationBackup");
  if (it == json.end())
    return false;

  auto&& load_it = it->find("load_from");
  return load_it != it->end() && load_it->is_number_integer();
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
  px = -1;
  py = -1;
  pz = -1;

  if (!config.json.contains("mpi"))
    return;

  const Configuration::json_t& mpi = config.json.at("mpi");

  auto get = [&mpi](const char* name, PetscInt& p) {
    if (auto it = mpi.find(name); it != mpi.end())
      p = it->get<PetscInt>();
  };

  get("da_processors_x", px);
  get("da_processors_y", py);
  get("da_processors_z", pz);
}
