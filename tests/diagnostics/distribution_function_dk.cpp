#include "petscmath.h"
#include "src/impls/drift_kinetic/diagnostic.h"
#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing \'DkDistributionFunction\' diagnostic for drift-kinetic      \n"
  "particles. Three guiding centers in the uniform magnetic field B=2   \n"
  "along z are binned in (v_parallel, mu_p) in their own cells. The     \n"
  "last one has v_parallel out of the axis range and should be omitted. \n"
  "Only one frame is allowed.                                           \n";

static constexpr PetscReal q = -1.0;
static constexpr PetscReal m = 2.0;
static constexpr PetscReal n = 1.0;
static constexpr PetscInt Np = 2;
static constexpr PetscReal B_value = 2.0;

const drift_kinetic::DkPhaseAxis v_parallel{
  .min = -1,
  .max = 1,
  .bins = 16,
};

const drift_kinetic::DkPhaseAxis mu_p{
  .min = 0,
  .max = 0.08,
  .bins = 4,
};

const std::vector<Point> points{
  Point{
    .r = Vector3R{2, 3, 4},
    .p = Vector3R{0.1, 0.2, 0.3},
  },
  Point{
    .r = Vector3R{5.5, 5, 5},
    .p = Vector3R{-0.3, 0.0, 0.4},
  },
  Point{
    .r = Vector3R{1, 1, 1},
    .p = Vector3R{0.0, 0.0, 5.0},
  },
};

void overwrite_config();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();

  drift_kinetic::Simulation simulation;
  PetscCall(simulation.initialize());

  auto& particles = *simulation.particles_[0];
  auto dir = get_out_dir(__FILE__) / "distribution_function";
  PetscInt max_frames = 1;

  auto&& diag = drift_kinetic::DkDistributionFunction::create(
    dir, particles, v_parallel, mu_p, 1, max_frames);

  PetscCall(diag->diagnose(0));
  PetscCall(diag->diagnose(1));
  PetscCall(diag->finalize());

  PetscCheck(!std::filesystem::exists(dir / "0001"), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Frames beyond \"max_frames\" should not be written");

  const PetscInt nv = v_parallel.bins;
  const PetscInt nmu = mu_p.bins;
  const PetscReal dv = (v_parallel.max - v_parallel.min) / nv;
  const PetscReal dmu = (mu_p.max - mu_p.min) / nmu;

  std::vector<float> written;
  PetscCall(read_floats(dir / "0000", geom_nx * geom_ny * geom_nz * nv * nmu, written));

  // Each point is loaded `Np` times with the weight `mpw`
  std::vector<PetscReal> expected(written.size(), 0.0);
  auto deposit = [&](const Vector3I& cell, PetscReal p_par, PetscReal mu)
  {
    PetscInt c = (cell[Z] * geom_ny + cell[Y]) * geom_nx + cell[X];
    PetscInt iv = FLOOR_STEP(p_par - v_parallel.min, dv);
    PetscInt imu = FLOOR_STEP(mu - mu_p.min, dmu);
    PetscInt g = (c * nv + iv) * nmu + imu;
    expected[g] += n / (dv * dmu);
  };

  // mu = m * p_perp^2 / (2B); the third point is out of the v_parallel range
  deposit({2, 3, 4}, 0.3, m * (POW2(0.1) + POW2(0.2)) / (2 * B_value));
  deposit({5, 5, 5}, 0.4, m * POW2(0.3) / (2 * B_value));

  for (std::size_t i = 0; i < written.size(); i++) {
    PetscCheck(equal_tol(written[i], expected[i], 1e-8), PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Distribution differs from the expected one at the index %zu: %g vs %g", i, (double)written[i], expected[i]);
  }

  PetscCall(simulation.finalize());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config()
{
  dx = dy = dz = 1.0;
  geom_nx = geom_ny = geom_nz = 8;
  geom_x = geom_y = geom_z = 8.0;

  dt = 1.0;
  geom_nt = 1;
  geom_t = 1.0;

  const char* sort_name = "particles";

  Configuration::array_t presets;

  presets.push_back({
    {"command", "SetField"},
    {"field", "B"},
    {"setter",
      {
        {"name", "SetUniformField"},
        {"value", {0.0, 0.0, B_value}},
      }},
  });

  for (const Point& point : points) {
    presets.push_back({
      // `PreciseCoordinate` loads `Np` copies
      {"command", "SetParticles"},
      {"particles", sort_name},
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

  Configuration::overwrite({
    {"Simulation", "drift_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
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
    {"Fields", {{"evolve", false}}},
    {
      "Particles",
      {{
        {"sort_name", sort_name},
        {"Np", Np},
        {"n", n},
        {"q", q},
        {"m", m},
      }},
    },
    {"Presets", presets},
  });
}
