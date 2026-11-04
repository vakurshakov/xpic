#include "src/diagnostics/distribution_moment.h"

#include "src/impls/basic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing \'DistributionMoment\' diagnostic for plain particles.         \n"
  "Two particles are deposited: one exactly at the grid node, another one \n"
  "halfway between two nodes. Each is spread by the sort's shape.   \n";

constexpr PetscReal q = -1.0;
constexpr PetscReal m = 2.0;
constexpr PetscReal n = 1.0;

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

PetscErrorCode check_moment(  //
  const interfaces::Particles& particles, const std::string& name, PetscInt dof);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();

  basic::Simulation simulation;
  PetscCall(simulation.initialize());

  auto& particles = *simulation.particles_[0];

  PetscCall(check_moment(particles, "density", 1));
  PetscCall(check_moment(particles, "current", 3));
  PetscCall(check_moment(particles, "momentum_flux", 6));
  PetscCall(check_moment(particles, "momentum_flux_diag", 3));

  PetscCall(simulation.finalize());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode check_moment(  //
  const interfaces::Particles& particles, const std::string& name, PetscInt dof)
{
  PetscFunctionBeginUser;
  Region region{
    .dim = dof > 1 ? 4 : 3,
    .dof = dof,
    .start = Vector4I{0, 0, 0, 0},
    .size = Vector4I{geom_nx, geom_ny, geom_nz, dof},
  };

  auto moment = moment_from_string(name);
  auto dir = get_out_dir(__FILE__) / name;

  auto&& diag = DistributionMoment::create(dir, particles, moment, region);
  PetscCall(diag->diagnose(0));
  PetscCall(diag->finalize());

  using Written = std::vector<float>;
  using Values = std::vector<PetscReal>;

  Written written;
  PetscCall(read_floats(dir / "0000", geom_nx * geom_ny * geom_nz * dof, written));

  Values expected(written.size(), 0.0);
  // Each particle is spread over the neighbouring cells by the shape, wrapping periodically
  const PetscInt R = (PetscInt)std::ceil(shape_radius);
  auto deposit = [&](const Point& point, const Values& values)
  {
    for (PetscInt dk = -R; dk <= R; dk++) {
      for (PetscInt dj = -R; dj <= R; dj++) {
        for (PetscInt di = -R; di <= R; di++) {
          const PetscInt cx = (PetscInt)std::floor(point.r[X]) + di;
          const PetscInt cy = (PetscInt)std::floor(point.r[Y]) + dj;
          const PetscInt cz = (PetscInt)std::floor(point.r[Z]) + dk;
          const PetscReal weight =             //
            shape_function(point.r[X] - cx) *  //
            shape_function(point.r[Y] - cy) *  //
            shape_function(point.r[Z] - cz);
          for (PetscInt c = 0; c < dof; c++) {
            PetscInt g = indexing::t_g(  //
              (cx + geom_nx) % geom_nx,  //
              (cy + geom_ny) % geom_ny,  //
              (cz + geom_nz) % geom_nz, c, dof);
            expected[g] += weight * n * values[c];
          }
        }
      }
    }
  };

  for (const Point& point : points)
    deposit(point, moment(particles, point));

  for (std::size_t i = 0; i < written.size(); i++) {
    PetscCheck(equal_tol(written[i], expected[i], 1e-7), PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Moment \"%s\" differs from the expected one at the index %zu: %g vs %g", name.c_str(), i, (double)written[i], expected[i]);
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
        {"Np", 1},
        {"n", n},
        {"q", q},
        {"m", m},
      }},
    },
    {"Presets", presets},
  });
}
