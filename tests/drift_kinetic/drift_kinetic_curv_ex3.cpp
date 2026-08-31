#include "src/impls/drift_kinetic/simulation.h"
#include "src/impls/eccapfim/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

/**
 * @brief Curvature drift comparison between "drift_kinetic" and "eccapfim".
 *
 * A prescribed azimuthal field of constant magnitude, B = B0 e_phi (circular
 * field lines around the z-axis, |B| = B0 everywhere). Then grad|B| = 0 (no
 * gradient drift) while rot b = z / r, so the parallel streaming yields a pure
 * curvature drift along z: V_kappa = (m v_par^2) / (q B0 R) z, where R is the
 * field-line (curvature) radius = distance of the guiding center from the axis.
 * This checks the interpolation of rot B; the field ~ 1/r is non-linear so the
 * discrete rot B is exact only to 2nd order in dx, so a large R/dx is used to
 * suppress the residual. Only a quarter of the field-line circle is covered:
 * the axis sits one margin in from the lower-left transverse corner and the box
 * spans the first quadrant (R + 2*margin) so the whole arc + gyration fit; this
 * keeps the box affordable while allowing a large R. A
 * single electron is seeded at the same coordinate for both runs (transferred
 * to the guiding center automatically for drift_kinetic). The domain is
 * elongated along the drift (z); x and y hold the field-line circle. Each run
 * writes its per-step coordinate trace into its own subdirectory:
 * <out>/{drift_kinetic,eccapfim}/temporal/particle_trace.txt.
 */

static constexpr char help[] = "Curvature drift comparison: drift_kinetic vs eccapfim.\n";

static constexpr PetscReal cell = 0.5;
static constexpr PetscReal B0_value = 1.0;  // constant |B|
static constexpr PetscReal R_value = 15.0;  // field-line (curvature) radius (large R -> small rot B error)
static constexpr PetscReal margin = 1;  // gyro-radius + interpolation stencil around the arc

// Only a quarter of the field-line circle is covered: the axis (|| z) sits one
// margin in from the lower-left transverse corner, and the box spans the first
// quadrant (R + 2*margin) so the whole quarter arc + gyration fit inside.
static constexpr PetscInt grid_nx =
  static_cast<PetscInt>((R_value + 2 * margin) / cell);
static constexpr PetscInt grid_ny = grid_nx;
static constexpr PetscInt grid_nz = 10;

static constexpr PetscReal total_time = 300.0;
static constexpr PetscReal dt_kinetic = 0.1;
static constexpr PetscReal dt_drift = 10.0;

static const Vector3R center(margin, margin, 0.0);
// Particle on the +x side of the circle: there e_phi = +y, so p0 = (v_perp, v_par, 0).
static const Vector3R r0(center[X] + R_value, center[Y], grid_nz * cell / 2);
static const Vector3R p0(0.2, 0.08, 0.0);

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
  geom_nx = grid_nx;
  geom_ny = grid_ny;
  geom_nz = grid_nz;
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
          {"setter", {
            {"name", "SetAzimuthalField"},
            {"value", B0_value},
            {"center", {center[X], center[Y], 0.0}},
          }},
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
