#include "src/impls/drift_kinetic/simulation.h"
#include "src/impls/eccapfim/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

/**
 * @brief ExB drift comparison between "drift_kinetic" and "eccapfim".
 *
 * A single electron is seeded at the same coordinate near one edge of a domain
 * elongated along the drift direction (x) and only 3x3 cells across (y, z).
 * Uniform fields E=0.01 (along y) and B=1 (along z) give an ExB drift
 * v=ExB/B^2=0.01 along x. Both runs span the same physical time, but the
 * kinetic step (dt=0.1) resolves the gyration while the drift-kinetic step
 * (dt=10) follows only the guiding center, doing 100x fewer steps. The shared
 * initial coordinate is transferred to the guiding center automatically for
 * the drift-kinetic run. Each run writes a per-step coordinate trace into its
 * own subdirectory: <out>/{drift_kinetic,eccapfim}/temporal/particle_trace.txt.
 */

static constexpr char help[] = "ExB drift comparison: drift_kinetic vs eccapfim.\n";

static constexpr PetscReal cell = 0.5;
static constexpr PetscReal E_value = 0.01;
static constexpr PetscReal B_value = 1.0;

static constexpr PetscReal total_time = 300.0;
static constexpr PetscReal dt_kinetic = 0.1;
static constexpr PetscReal dt_drift = 10.0;

static const Vector3R r0(0.5, 0.55, 0.75);
static const Vector3R p0(0.2, 0.0, 0.0);

void overwrite_config(
  const std::string& simulation, const std::string& out_dir, PetscReal step);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  const std::filesystem::path base = get_out_dir(__FILE__);
  const std::string dk_dir = (base / "drift_kinetic").string();
  const std::string kin_dir = (base / "eccapfim").string();

  overwrite_config("drift_kinetic", dk_dir, dt_drift);
  std::filesystem::create_directories(dk_dir);
  Configuration::save(dk_dir);
  {
    drift_kinetic::Simulation simulation;
    PetscCall(simulation.initialize());
    PetscCall(simulation.calculate());
    PetscCall(simulation.finalize());
  }

  overwrite_config("eccapfim", kin_dir, dt_kinetic);
  std::filesystem::create_directories(kin_dir);
  Configuration::save(kin_dir);
  {
    eccapfim::Simulation simulation;
    PetscCall(simulation.initialize());
    PetscCall(simulation.calculate());
    PetscCall(simulation.finalize());
  }

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config(
  const std::string& simulation, const std::string& out_dir, PetscReal step)
{
  dx = dy = dz = cell;
  geom_nx = 10;
  geom_ny = geom_nz = 3;
  geom_x = geom_nx * dx;
  geom_y = geom_ny * dy;
  geom_z = geom_nz * dz;

  dt = step;
  geom_nt = static_cast<PetscInt>(std::llround(total_time / dt));
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", simulation},
    {"OutputDirectory", out_dir},
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
        {"n", +1.0},
        {"q", -1.0},
        {"m", +1.0},
        {"T", +0.1},
      }},
    },
    {
      "Presets",
      {
        {
          {"command", "SetMagneticField"},
          {"field", "B0"},
          {"field_axpy", "B"},
          {"setter", {{"name", "SetUniformField"}, {"value", {0.0, 0.0, B_value}}}},
        },
        {
          {"command", "SetElectricField"},
          {"setter", {{"name", "SetUniformField"}, {"value", {0.0, E_value, 0.0}}}},
        },
        {
          {"command", "SetParticles"},
          {"particles", "electrons"},
          {"coordinate", {{"name", "PreciseCoordinate"}, {"value", {r0[X], r0[Y], r0[Z]}}}},
          {"momentum", {{"name", "PreciseMomentum"}, {"value", {p0[X], p0[Y], p0[Z]}}}},
        },
      },
    },
  });
}
