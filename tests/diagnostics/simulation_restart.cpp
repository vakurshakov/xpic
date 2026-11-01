#include "src/interfaces/simulation.h"

#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing that a run restarted from \"SimulationBackup\" reproduces the     \n"
  "continuous one. The simulation (`-restart_simulation`) is run for `N`     \n"
  "steps without a stop, then, for every `k`, it is run for `k` steps with a \n"
  "backup, loaded from it and continued up to `N`. Fields and particles at   \n"
  "`N` must be the same up to the round-off and the summation order.        \n";

static constexpr PetscInt N = 8;
static constexpr PetscInt k = 2;

struct State {
  std::vector<PetscReal> E;
  std::vector<PetscReal> B;
  Snapshot particles;
};

static const std::vector<Point> points{
  Point{.r = Vector3R{2, 3, 4}, .p = Vector3R{0.1, 0.2, 0.3}},
  Point{.r = Vector3R{5.5, 5, 5}, .p = Vector3R{-0.3, 0, 0.4}},
  Point{.r = Vector3R{7.5, 0.5, 6.5}, .p = Vector3R{0.2, -0.1, -0.4}},
  Point{.r = Vector3R{3.5, 6.5, 1.5}, .p = Vector3R{-0.2, 0.3, 0.1}},
};

static PetscErrorCode run(const std::string& simulation_name, PetscInt nt,
  const Configuration::json_t& backup, std::string_view subdir, State* state);

static PetscErrorCode compare(const State& expected, const State& actual);

static PetscErrorCode compare_tables(std::string_view subdir);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  char name[PETSC_MAX_PATH_LEN] = "";
  PetscCall(PetscOptionsGetString(NULL, NULL, "-restart_simulation", name, sizeof(name), NULL));

  // The tables of other simulations from the shared directory must not be compared
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0) {
    std::filesystem::remove_all(get_out_dir(__FILE__));
  }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  State continuous;
  PetscCall(run(name, N, {}, "continuous", &continuous));

  for (PetscInt t = k; t < N; t += k) {
    std::string subdir = std::format("restart_{}", t);

    PetscCall(run(name, t, {{"diagnose_period", (PetscReal)t}}, subdir, NULL));

    State restarted;
    PetscCall(run(name, N, {{"diagnose_period", (PetscReal)t}, {"load_from", t}}, subdir, &restarted));
    PetscCall(compare(continuous, restarted));
    PetscCall(compare_tables(subdir));
  }

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode run(const std::string& simulation_name, PetscInt nt,
  const Configuration::json_t& backup, std::string_view subdir, State* state)
{
  PetscFunctionBeginUser;
  dx = dy = dz = 1.0;
  geom_nx = geom_ny = geom_nz = 8;
  geom_x = geom_y = geom_z = 8.0;

  dt = 1.0;
  geom_nt = nt;
  geom_t = nt * dt;

  Configuration::array_t presets;

  presets.push_back({
    {"command", "SetField"},
    {"field", "B"},
    {"setter",
      {
        {"name", "SetUniformField"},
        {"value", {0, 0, 1}},
      }},
  });

  for (const Point& point : points) {
    presets.push_back({
      {"command", "SetParticles"},
      {"particles", "electrons"},
      {"coordinate",
        {
          {"name", "PreciseCoordinate"},
          {"value", {point.r[X], point.r[Y], point.r[Z]}},
        }},
      {"momentum",
        {
          {"name", "PreciseMomentum"},
          {"value", {point.p[X], point.p[Y], point.p[Z]}},
        }},
    });
  }

  Configuration::json_t json{
    {"Simulation", simulation_name},
    {"OutputDirectory", get_out_dir(__FILE__) / subdir},
    {"DriftKineticPush", {{"coord_is_gc", true}}},
    {
      "Geometry",
      {
        {"x", geom_x},
        {"y", geom_y},
        {"z", geom_z},
        {"t", geom_t},
        {"dx", dx},
        {"dy", dy},
        {"dz", dz},
        {"dt", dt},
        {"diagnose_period", dt},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "electrons"},
        {"Np", 1},
        {"n", 1},
        {"q", -1},
        {"m", 1},
      }},
    },
    {"Presets", presets},
  };

  if (!backup.is_null())
    json["SimulationBackup"] = backup;

  Configuration::overwrite(std::move(json));

  auto simulation = build_simulation();
  PetscCall(simulation->initialize());
  PetscCall(simulation->calculate());

  if (state) {
    for (auto&& [vec, out] : {                   //
           std::pair{simulation->E, &state->E},  //
           std::pair{simulation->B, &state->B}}) {
      PetscInt size;
      const PetscReal* array;
      PetscCall(VecGetLocalSize(vec, &size));
      PetscCall(VecGetArrayRead(vec, &array));
      out->assign(array, array + size);
      PetscCall(VecRestoreArrayRead(vec, &array));
    }
    PetscCall(take_snapshot(simulation->particles_[0]->storage, state->particles));
  }

  PetscCall(simulation->finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode compare(const State& expected, const State& actual)
{
  PetscFunctionBeginUser;
  auto close = [](const std::vector<PetscReal>& a, const std::vector<PetscReal>& b)
  {
    return std::ranges::equal(a, b,
      [](PetscReal x, PetscReal y)
      {
        return equal_rel(x, y, PETSC_MACHINE_EPSILON, 0);
      });
  };

  PetscCheck(close(expected.E, actual.E), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Electric field of the restarted run differs from the continuous one");

  PetscCheck(close(expected.B, actual.B), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Magnetic field of the restarted run differs from the continuous one");

  PetscCheck(expected.particles.size() == actual.particles.size(), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Number of particles has changed after the restart: %zu -> %zu", expected.particles.size(), actual.particles.size());

  for (size_t i = 0; i < expected.particles.size(); i++) {
    PetscCheck(close(expected.particles[i], actual.particles[i]), PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle %zu of the restarted run differs from the continuous one", i);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Every temporal table of the restarted run must be the continuous one,
/// including the rows saved before the backup, which are loaded with it.
static PetscErrorCode compare_tables(std::string_view subdir)
{
  PetscFunctionBeginUser;
  auto read = [](const std::filesystem::path& path)
  {
    std::ifstream ifs(path);
    std::string header;
    std::getline(ifs, header);
    std::vector<PetscReal> values;
    for (PetscReal v; ifs >> v;)
      values.push_back(v);
    return std::pair{header, values};
  };

  for (auto&& entry : std::filesystem::directory_iterator(
         get_out_dir(__FILE__) / "continuous" / "temporal")) {
    auto name = entry.path().filename();
    auto [e_header, e_values] = read(entry.path());
    auto [o_header, o_values] =
      read(get_out_dir(__FILE__) / subdir / "temporal" / name);

    PetscCheck(e_header == o_header, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Header of \"%s\" differs after the restart", name.c_str());

    PetscCheck(e_values.size() == o_values.size(), PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Size of \"%s\" differs after the restart: %zu -> %zu values", name.c_str(), e_values.size(), o_values.size());

    auto check_conservation =
      [&](std::string_view logname, std::string_view colname, PetscReal tol)
    {
      PetscFunctionBeginUser;
      std::istringstream titles(o_header);
      std::vector<std::string> columns{
        std::istream_iterator<std::string>(titles), {}};
      size_t col = std::ranges::find(columns, colname) - columns.begin();
      PetscCheck(col < columns.size(), PETSC_COMM_WORLD, PETSC_ERR_USER, "No \"%s\" column", colname.data());

      for (size_t i = col; i < o_values.size(); i += columns.size()) {
        PetscCheck(std::abs(o_values[i]) < tol, PETSC_COMM_WORLD, PETSC_ERR_USER,
          "%s is not conserved after the restart, row %zu: %.6e", logname.data(), i / columns.size(), o_values[i]);
      }
      PetscFunctionReturn(PETSC_SUCCESS);
    };

    // Energy and charge is conserved to the round-off, also on the rows computed after the restart

    /// @todo to enable this, we should properly calculate the kinetic energy on the sort and map the output to default `energy_conservation.txt`
    /// if (name == "energy_conservation.txt") {
    ///   PetscCall(check_conservation("Energy", "dE+dB+dK", 1e-7));
    /// }
    /// else

    if (name == "charge_conservation.txt") {
      PetscCall(check_conservation("Charge", "N2dQ_tot", 1e-10));
    }

    for (size_t i = 0; i < e_values.size(); i++) {
      PetscCheck(equal_rel(e_values[i], o_values[i], PETSC_SMALL, 1e-10), PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Value %zu of \"%s\" differs after the restart: expected %.6e, got %.6e", i, name.c_str(), e_values[i], o_values[i]);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
