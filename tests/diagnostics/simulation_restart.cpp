#include "src/interfaces/simulation.h"

#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing that a run restarted from \"SimulationBackup\" reproduces the     \n"
  "continuous one. The simulation (`-restart_simulation`) is run for `N`     \n"
  "steps without a stop, then, for every `k`, it is run for `k` steps with a \n"
  "backup, loaded from it and continued up to `N`. Fields and particles at   \n"
  "`N` must be the same up to the round-off and the summation order.         "
  "\n";

static constexpr PetscInt N = 8;
static constexpr PetscInt k = 2;
static constexpr PetscReal tolerance = 1e-8;

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

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  char name[PETSC_MAX_PATH_LEN] = "";
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-restart_simulation", name, sizeof(name), nullptr));

  State continuous;
  PetscCall(run(name, N, nullptr, "continuous", &continuous));

  for (PetscInt t = k; t < N; t += k) {
    std::string subdir = std::format("restart_{}", t);

    PetscCall(run(name, t, {{"diagnose_period", (PetscReal)t}}, subdir, nullptr));

    State restarted;
    PetscCall(run(name, N, {{"diagnose_period", (PetscReal)t}, {"load_from", t}}, subdir, &restarted));
    PetscCall(compare(continuous, restarted));
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
    {"setter", {{"name", "SetUniformField"}, {"value", {0, 0, 1}}}},
  });

  for (const Point& point : points) {
    presets.push_back({
      {"command", "SetParticles"},
      {"particles", "electrons"},
      {"coordinate",
        {{"name", "PreciseCoordinate"},
          {"value", {point.r[X], point.r[Y], point.r[Z]}}}},
      {"momentum",
        {{"name", "PreciseMomentum"},
          {"value", {point.p[X], point.p[Y], point.p[Z]}}}},
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
    for (auto&& [vec, out] :
      {std::pair{simulation->E, &state->E}, {simulation->B, &state->B}}) {
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
        return equal_rel(x, y, tolerance, tolerance);
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
