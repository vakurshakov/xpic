#include "src/impls/drift_kinetic/diagnostic.h"
#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing \'DkDistributionMoment\' diagnostic for drift-kinetic particles. \n"
  "Two guiding centers in the uniform magnetic field B=2 along z are        \n"
  "deposited: one exactly at the grid node, which should land entirely in   \n"
  "a single cell, and another one halfway between two nodes, which should   \n"
  "be split equally.                                                        \n";

static constexpr PetscReal q = -1.0;
static constexpr PetscReal m = 2.0;
static constexpr PetscReal n = 1.0;
static constexpr PetscInt Np = 2;
static constexpr PetscReal B_value = 2.0;

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

using Written = std::vector<float>;
using Values = std::vector<PetscReal>;

void overwrite_config();

PetscErrorCode check_moment(const drift_kinetic::Particles& particles,
  const std::string& name, PetscInt dof);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();

  drift_kinetic::Simulation simulation;
  PetscCall(simulation.initialize());

  auto& particles = *simulation.particles_[0];

  PetscCall(check_moment(particles, "density", 1));
  PetscCall(check_moment(particles, "velocity", 3));
  PetscCall(check_moment(particles, "velocity_parallel", 1));
  PetscCall(check_moment(particles, "temperature_parallel", 1));
  PetscCall(check_moment(particles, "temperature_perp", 1));
  PetscCall(check_moment(particles, "momentum_flux_diag", 3));

  PetscCall(simulation.finalize());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Expected moments, written independently from the diagnostic getters.
Values expected_moment(const std::string& name, const Point& point)
{
  const auto& p = point.p;
  const PetscReal p_perp2 = POW2(p[X]) + POW2(p[Y]);
  if (name == "density")
    return {1.0};
  if (name == "velocity")
    return {p[X], p[Y], p[Z]};
  if (name == "velocity_parallel")
    return {p[Z]};
  if (name == "temperature_parallel")
    return {m * POW2(p[Z])};
  if (name == "temperature_perp")
    return {m * p_perp2};  // 2 * mu * B, mu = m * p_perp^2 / (2B)
  if (name == "momentum_flux_diag")
    return {m * POW2(p[X]), m * POW2(p[Y]), m * POW2(p[Z])};
  throw std::runtime_error("Unexpected moment " + name);
}

PetscErrorCode check_moment(const drift_kinetic::Particles& particles,
  const std::string& name, PetscInt dof)
{
  PetscFunctionBeginUser;
  Region region{
    .dim = dof > 1 ? 4 : 3,
    .dof = dof,
    .start = Vector4I{0, 0, 0, 0},
    .size = Vector4I{geom_nx, geom_ny, geom_nz, dof},
  };

  auto out_dir = get_out_dir(__FILE__) / "moments" / name;

  // Density has no dependency on B, so it is also checked through the plain `Moment`.
  auto&& diag = (name == "density")
    ? drift_kinetic::DkDistributionMoment::create(
        out_dir, particles, moment_from_string(name), region)
    : drift_kinetic::DkDistributionMoment::create(
        out_dir, particles, drift_kinetic::dk_moment_from_string(name), region);
  PetscCall(diag->diagnose(0));
  PetscCall(diag->finalize());

  Written written;
  PetscCall(read_floats(out_dir / "0000", geom_nx * geom_ny * geom_nz * dof, written));

  Values expected(written.size(), 0.0);
  auto deposit = [&](const Vector3I& cell, PetscReal weight, const Values& values)
  {
    for (PetscInt c = 0; c < dof; c++) {
      PetscInt g = indexing::t_g(cell[X], cell[Y], cell[Z], c, dof);
      expected[g] += weight * n * values[c];
    }
  };

  deposit({2, 3, 4}, 1.0, expected_moment(name, points[0]));
  deposit({5, 5, 5}, 0.5, expected_moment(name, points[1]));
  deposit({6, 5, 5}, 0.5, expected_moment(name, points[1]));

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
