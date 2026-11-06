#include "src/impls/drift_kinetic/simulation.h"
#include "src/impls/eccapfim/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Bounce motion in a mirror, \"drift_kinetic\" and \"eccapfim\".            \n"
  "                                                                          \n"
  "Paraxial parabolic mirror along z, B_z = B0 (1 + kappa s^2), s = z - z_c, \n"
  "with the divergence-free radial part B_x,y = -B0 kappa s (x,y - c_x,y).   \n"
  "On the axis |B| = B_z, so the mirror force -mu dB/dz makes the guiding    \n"
  "center a harmonic oscillator with omega_b = sqrt(2 mu B0 kappa / m). The  \n"
  "drift-kinetic step reduces to the implicit midpoint rule: the energy      \n"
  "m v_par^2 / 2 + mu B is conserved exactly and the bounce frequency is     \n"
  "omega_h = (2 / dt) atan(omega_b dt / 2). The electron is seeded one Larmor\n"
  "radius off the axis, so that its guiding center lies on it. All runs span \n"
  "the same physical time; the kinetic step (dt=0.1) resolves the gyration,  \n"
  "the drift-kinetic one is scanned over dt = 10..0.625 to show the 2nd      \n"
  "order of the bounce frequency. Each run writes its per-step coordinate    \n"
  "trace into its own subdirectory:                                          \n"
  "  <out>/{drift_kinetic_dt<dt>,eccapfim}/temporal/particle_trace.txt.     \n";

static constexpr PetscReal cell = 0.5;
static constexpr PetscReal B0_value = 1.0;
static constexpr PetscReal curvature = 0.05;

static constexpr PetscReal total_time = 900.0;
static constexpr PetscReal dt_kinetic = 0.1;
static constexpr PetscReal dt_drift[] = {10.0, 5.0, 2.5, 1.25, 0.625};

static const Vector3R center(2.0, 2.0, 6.0);
static const Vector3R p0(0.4 / 3.0, 0.0, 0.1);

static constexpr PetscReal s0 = 0.4;
static const PetscReal rho0 = p0[X] / (B0_value * (1.0 + curvature * s0 * s0));
static const Vector3R r0(center[X], center[Y] - rho0, center[Z] + s0);

void overwrite_config(
  const std::string& simulation, const std::string& out_dir, PetscReal step);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  const std::filesystem::path base = get_out_dir(__FILE__);
  const std::string dir_kinetic = (base / "eccapfim").string();

  overwrite_config("eccapfim", dir_kinetic, dt_kinetic);
  Configuration::save(dir_kinetic);
  {
    eccapfim::Simulation simulation;
    PetscCall(simulation.initialize());
    PetscCall(simulation.calculate());
    PetscCall(simulation.finalize());
  }

  for (PetscReal step : dt_drift) {
    const std::string dir_drift =
      (base / std::format("drift_kinetic_dt{}", step)).string();

    overwrite_config("drift_kinetic", dir_drift, step);
    Configuration::save(dir_drift);
    {
      drift_kinetic::Simulation simulation;
      PetscCall(simulation.initialize());
      PetscCall(simulation.calculate());
      PetscCall(simulation.finalize());
    }
  }

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config(
  const std::string& simulation, const std::string& out_dir, PetscReal step)
{
  dx = dy = dz = cell;
  geom_nx = geom_ny = 8;
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
              {"name", "SetMirrorField"},
              {"value", B0_value},
              {"curvature", curvature},
              {"center", {center[X], center[Y], center[Z]}},
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
