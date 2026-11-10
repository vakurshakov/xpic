#include "src/impls/hybrid_drift_kinetic/simulation.h"
#include "src/diagnostics/energy.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr PetscReal mass = 2;
static constexpr PetscReal density = 3;
static constexpr PetscInt particles_per_cell = 2;

static const Point points[]{
  {.r = {1.5, 2.5, 3.5}, .p = {0.2, -0.1, 0.3}},
  {.r = {6.5, 5.5, 4.5}, .p = {-0.4, 0.2, -0.1}},
};

struct EnergyProbe : Energy {
  using Energy::Energy;
  using Energy::calculate_kinetic;
  using Energy::K;
};

static void overwrite_config();
static PetscErrorCode check_energy(
  const interfaces::Particles& sort, PetscReal perpendicular_factor);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
  overwrite_config();

  hybrid_drift_kinetic::Simulation simulation;
  PetscCall(simulation.initialize());
  auto& sorts = simulation.interfaces::Simulation::particles_;

  for (const auto& sort : sorts) {
    PetscCheck(sort->kinetic_energy_local() == 0, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "An empty sort must have zero kinetic energy");

    std::vector<Point> local;
    for (const auto& point : points)
      if (sort->is_local(point))
        local.push_back(point);
    PetscCall(sort->ingest(local.data(), local.size()));
    PetscCall(check_energy(*sort, 1));
  }

  PetscCall(VecScale(simulation.B, 2));
  PetscCall(check_energy(*simulation.kinetic_particles_[0], 1));
  PetscCall(check_energy(*simulation.particles_[0], 2));

  EnergyProbe diagnostic(simulation);
  PetscCall(diagnostic.calculate_kinetic());
  // Initial sum |v|^2 is 0.35; doubling B adds the perpendicular sum 0.25.
  const PetscReal weight = 0.5 * mass * density / particles_per_cell;
  PetscCheck(equal_rel(diagnostic.K[0], weight * 0.35, 1e-12, 1e-14) &&
    equal_rel(diagnostic.K[1], weight * 0.60, 1e-12, 1e-14),
    PETSC_COMM_WORLD, PETSC_ERR_USER,
    "The general energy diagnostic must use each sort's energy and reduce it globally");

  PetscCall(simulation.finalize());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode check_energy(
  const interfaces::Particles& sort, PetscReal perpendicular_factor)
{
  PetscFunctionBeginUser;
  PetscReal expected = 0;
  for (const auto& point : points) {
    if (!sort.is_local(point))
      continue;
    expected += POW2(point.p[Z]) + perpendicular_factor *
      (POW2(point.p[X]) + POW2(point.p[Y]));
  }
  expected *= 0.5 * mass * density / particles_per_cell;

  const PetscReal actual = sort.kinetic_energy_local();
  PetscCheck(equal_rel(actual, expected, 1e-12, 1e-14), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Local kinetic energy of %s differs: %g vs %g", sort.parameters.sort_name.c_str(), actual, expected);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void overwrite_config()
{
  dx = dy = dz = 1;
  geom_nx = geom_ny = geom_nz = 8;
  geom_x = geom_y = geom_z = 8;
  dt = 1;
  geom_nt = 0;
  geom_t = 0;

  Configuration::overwrite({
    {"Simulation", "hybrid_drift_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {"DriftKineticPush", {{"coord_is_gc", true}}},
    {"Geometry",
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
      }},
    {"Fields", {{"evolve", false}}},
    {"Particles",
      {
        {{"sort_name", "ions"}, {"model", "kinetic"},
          {"Np", particles_per_cell}, {"n", density}, {"q", 1}, {"m", mass}},
        {{"sort_name", "electrons"}, {"model", "drift_kinetic"},
          {"Np", particles_per_cell}, {"n", density}, {"q", -1}, {"m", mass}},
      }},
    {"Presets",
      {
        {{"command", "SetField"}, {"field", "B"},
          {"setter", {{"name", "SetUniformField"}, {"value", {0, 0, 2}}}}},
      }},
  });
}
