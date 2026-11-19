#include "src/impls/drift_kinetic/simulation.h"
#include "src/impls/eccapfim/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

#if defined(NONLIN_GRADB_ALIGNED) == defined(NONLIN_GRADB_OBLIQUE)
  #error "Define exactly one of NONLIN_GRADB_ALIGNED or NONLIN_GRADB_OBLIQUE"
#endif

static constexpr char help[] =
  "Spatial convergence of non-linear grad-B drift against kinetic motion.\n"
  "A periodic magnetic magnitude gives a non-uniform gradient.\n"
  "CMake selects aligned or oblique fields; both use one kinetic reference\n"
  "and drift-kinetic runs on 16, 32, 64 and 128 cells per wavelength.\n";

static constexpr PetscReal B0_value = 1.0;
static constexpr PetscReal modulation = 0.2;
static constexpr PetscReal length = 20.0;
static constexpr PetscReal cell = 0.5;
static constexpr PetscReal total_time = 300.0;
static constexpr PetscReal dt_kinetic = 0.1;
static constexpr PetscReal dt_drift = 10.0;
static constexpr PetscInt n_kinetic = 64;
static constexpr PetscInt n_drift[] = {16, 32, 64, 128};

#if defined(NONLIN_GRADB_ALIGNED)
static constexpr char case_name[] = "aligned";
static const Vector3R b0(0.0, 0.0, 1.0);
static const Vector3R wave_number(1.0, 0.0, 0.0);
static const Vector3R r0(length / 6, 1.0, 0.75);
static const Vector3R p0(0.5, 0.0, 0.0);
#else
static constexpr char case_name[] = "oblique";
static const Vector3R b0(0.6 * M_SQRT1_2, -0.6 * M_SQRT1_2, 0.8);
static const Vector3R wave_number(1.0, 1.0, 0.0);
static const Vector3R r0(length / 12, length / 12, 0.75);
static const Vector3R p0(0.5 * M_SQRT1_2, 0.5 * M_SQRT1_2, 0.0);
#endif

void overwrite_config(const std::string& simulation, const std::string& out_dir,
  PetscReal step, PetscInt n);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  const std::filesystem::path base = get_out_dir(__FILE__) / case_name;
  const std::string dir_kinetic = (base / "eccapfim").string();

  overwrite_config("eccapfim", dir_kinetic, dt_kinetic, n_kinetic);
  Configuration::save(dir_kinetic);
  {
    eccapfim::Simulation simulation;
    PetscCall(simulation.initialize());
    PetscCall(simulation.calculate());
    PetscCall(simulation.finalize());
  }

  for (PetscInt n : n_drift) {
    const std::string dir_drift =
      (base / std::format("drift_kinetic_n{}", n)).string();

    overwrite_config("drift_kinetic", dir_drift, dt_drift, n);
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

void overwrite_config(const std::string& simulation, const std::string& out_dir,
  PetscReal step, PetscInt n)
{
  dx = length / n;
#if defined(NONLIN_GRADB_ALIGNED)
  dy = dz = cell;
  geom_nx = n;
  geom_ny = 16;
  geom_nz = 4;
#else
  dy = dx;
  dz = cell;
  geom_nx = geom_ny = n;
  geom_nz = 8;
#endif
  geom_x = geom_nx * dx;
  geom_y = geom_ny * dy;
  geom_z = geom_nz * dz;

  dt = step;
  geom_nt = ROUND_STEP(total_time, dt);
  geom_t = geom_nt * dt;

  const Vector3R B = B0_value * b0;
  const Vector3R dB = modulation * B;

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
              {"value", {B[X], B[Y], B[Z]}},
            }},
        },
        {
          {"command", "SetField"},
          {"field", "B"},
          {"setter",
            {
              {"name", "SetGeneralCosineField"},
              {"amplitude", {dB[X], dB[Y], dB[Z]}},
              {"wave_number", {wave_number[X], wave_number[Y], wave_number[Z]}},
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
