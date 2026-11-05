#include "src/impls/drift_kinetic/simulation.h"
#include "src/impls/eccapfim/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Parallel acceleration comparison, \"drift_kinetic\" and \"eccapfim\".     \n"
  "                                                                          \n"
  "Uniform fields B = 1 and E = 4e-4 along z. There is no drift; the guiding \n"
  "center moves along the field line with the constant acceleration         \n"
  "a = qE/m, so z(t) = z0 + v0 t + a t^2 / 2. The initial v_par = 0.04 is    \n"
  "reversed at t = 100. The implicit midpoint step is exact for a constant   \n"
  "force, so the drift-kinetic run (dt=10) must reproduce z(t), v_par(t) and \n"
  "the work m dv_par^2 / 2 = qE dz to round-off; the kinetic run (dt=0.1)    \n"
  "also resolves the gyration. A single electron is seeded at the same       \n"
  "coordinate for both runs. Each run writes its per-step coordinate trace   \n"
  "into its own subdirectory:                                                \n"
  "  <out>/{drift_kinetic,eccapfim}/temporal/particle_trace.txt.            \n";

static constexpr PetscReal cell = 0.5;
static constexpr PetscReal E_value = 4e-4;
static constexpr PetscReal B_value = 1.0;

static constexpr PetscReal total_time = 300.0;
static constexpr PetscReal dt_kinetic = 0.1;
static constexpr PetscReal dt_drift = 10.0;

static const Vector3R r0(0.5, 0.55, 8.0);
static const Vector3R p0(0.2, 0.0, 0.04);

void overwrite_config(
  const std::string& simulation, const std::string& out_dir, PetscReal step);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  const std::filesystem::path base = get_out_dir(__FILE__);
  const std::string dir_kinetic = (base / "eccapfim").string();
  const std::string dir_drift = (base / "drift_kinetic").string();

  overwrite_config("eccapfim", dir_kinetic, dt_kinetic);
  Configuration::save(dir_kinetic);
  {
    eccapfim::Simulation simulation;
    PetscCall(simulation.initialize());
    PetscCall(simulation.calculate());
    PetscCall(simulation.finalize());
  }

  overwrite_config("drift_kinetic", dir_drift, dt_drift);
  Configuration::save(dir_drift);
  {
    drift_kinetic::Simulation simulation;
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
  geom_nx = geom_ny = 4;
  geom_nz = 24;
  geom_x = geom_nx * dx;
  geom_y = geom_ny * dy;
  geom_z = geom_nz * dz;

  dt = step;
  geom_nt = ROUND_STEP(total_time, dt);
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
    {"Fields", {{"evolve", false}}},
    {
      "Particles",
      {
        {
          {"sort_name", "electrons"},
          {"Np", 1},
          {"n", +1.0},
          {"q", -1.0},
          {"m", +1.0},
          {"T", +0.1},
        },
      },
    },
    {
      "Presets",
      {
        {
          {"command", "SetField"},
          {"field", "B"},
          {"setter",
            {
              {"name", "SetUniformField"},
              {"value", {0.0, 0.0, B_value}},
            }},
        },
        {
          {"command", "SetField"},
          {"field", "E"},
          {"setter",
            {
              {"name", "SetUniformField"},
              {"value", {0.0, 0.0, E_value}},
            }},
        },
        {
          {"command", "SetParticles"},
          {"particles", "electrons"},
          {"coordinate",
            {
              {"name", "PreciseCoordinate"},
              {"value", {r0[X], r0[Y], r0[Z]}},
            }},
          {"momentum",
            {
              {"name", "PreciseMomentum"},
              {"value", {p0[X], p0[Y], p0[Z]}},
            }},
        },
      },
    },
    {
      "Diagnostics",
      {
        {
          {"diagnostic", "ParticleTrace"},
          {"particles", "electrons"},
        },
      },
    },
  });
}
