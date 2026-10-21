#include "src/diagnostics/distribution_function.h"

#include "src/impls/basic/simulation.h"
#include "src/utils/configuration.h"
#include "src/utils/utils.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing \'DistributionFunction\' diagnostic for plain particles. \n"
  "Two particles are placed in different cells, each one should fall\n"
  "into a single velocity bin. The geometry tester is then used to  \n"
  "leave only the particle from the cell with x < 4.                \n";

constexpr PetscReal q = -1.0;
constexpr PetscReal m = 2.0;
constexpr PetscReal n = 1.0;
constexpr PetscInt Np = 2;

constexpr PetscReal dv = 0.1;
constexpr PetscInt nv = (PetscInt)(2.0 / dv);  // bins in [-1, 1)

const std::vector<Point> points{
  Point{
    .r = Vector3R{2, 3, 4},
    .p = Vector3R{0.1, 0.2, 0.3},
  },
  Point{
    .r = Vector3R{5.5, 5, 5},
    .p = Vector3R{-0.3, 0.0, 0.4},
  },
};

void overwrite_config();

PetscErrorCode check_distribution(  //
  const interfaces::Particles& particles, const std::string& name,
  const Tester& within_geom, const std::vector<Point>& expected_points);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();

  basic::Simulation simulation;
  PetscCall(simulation.initialize());

  auto& particles = *simulation.particles_[0];

  auto all = [](const Vector3R&)
  {
    return true;
  };
  PetscCall(check_distribution(particles, "all", all, points));

  auto x_lt_4 = [](const Vector3R& r)
  {
    return r[X] < 4;
  };
  PetscCall(check_distribution(particles, "x_lt_4", x_lt_4, {points[0]}));

  PetscCall(simulation.finalize());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_distribution(  //
  const interfaces::Particles& particles, const std::string& name,
  const Tester& within_geom, const std::vector<Point>& expected_points)
{
  PetscFunctionBeginUser;
  Region xreg{
    .dim = 4,
    .dof = 1,
    .start = Vector4I{0, 0, 0, 0},
    .size = Vector4I{geom_nx, geom_ny, geom_nz, 1},
  };

  DistributionFunction::VelocityRegion vreg{
    .vx_max = +1,
    .vy_max = +1,
    .vx_min = -1,
    .vy_min = -1,
    .dvx = dv,
    .dvy = dv,
  };

  auto projector = projector_from_string("vx_vy");
  auto dir = get_out_dir(__FILE__) / name;

  auto&& diag = DistributionFunction::create(
    dir, particles, projector, within_geom, xreg, vreg);

  PetscCall(diag->diagnose(0));
  PetscCall(diag->finalize());

  std::vector<float> written;
  PetscCall(read_floats(dir / "0000", nv * nv, written));

  std::vector<PetscReal> expected(written.size(), 0.0);
  for (const Point& point : expected_points) {
    auto [vx, vy] = projector(point);

    auto vgx = ROUND_STEP(vx, dv) + nv / 2;
    auto vgy = ROUND_STEP(vy, dv) + nv / 2;
    auto g = indexing::petsc_index(vgx, vgy, 0, 0, nv, nv, 1, 1);

    // Each point is loaded `Np` times with the weight `mpw`
    expected[g] += n;
  }

  for (std::size_t i = 0; i < written.size(); i++) {
    PetscCheck(equal_tol(written[i], expected[i], 1e-8), PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Distribution \"%s\" differs from the expected one at the index %zu: %g vs %g", name.c_str(), i, (double)written[i], expected[i]);
  }
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
    {"Simulation", "basic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
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
