#include "src/impls/drift_kinetic/simulation.h"
#include "src/impls/eccapfim/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

/**
 * @brief Grad-B drift comparison between "drift_kinetic" and "eccapfim".
 *
 * Straight field lines with a varying magnitude, B = B0 (1 + (x - x0)/L) z (so
 * |B| = B0 at the particle's x0), E = 0, v_par = 0. Direction b = z is constant
 * so rot b = 0 (no curvature), leaving
 * only the gradient drift V_gradB = (mu/q) (b x grad|B|) / B = mu (B0/L)/(qB) y.
 * The field is linear in x, so the discrete gradient (B_{i+1}-B_i)/dx is exactly
 * B0/L and the drift is reproduced exactly; a non-linear B(x) would converge at
 * 2nd order in dx. A single electron is seeded at the same coordinate for both
 * runs (transferred to the guiding center automatically for drift_kinetic). The
 * domain is elongated along the drift (y); x carries the gradient, z is 3 cells.
 * Each run writes its per-step coordinate trace into its own subdirectory:
 *   <out>/{drift_kinetic,eccapfim}/temporal/particle_trace.txt.
 */

static constexpr char help[] = "Grad-B drift comparison: drift_kinetic vs eccapfim.\n";

static constexpr PetscReal cell = 0.5;
static constexpr PetscReal B0_value = 1.0;
static constexpr PetscReal L_value = 10.0;

static constexpr PetscReal total_time = 300.0;
static constexpr PetscReal dt_kinetic = 0.1;
static constexpr PetscReal dt_drift = 10.0;

static const Vector3R r0(2.5, 5.0, 0.75);
static const Vector3R p0(0.5, 0.0, 0.0);

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
  geom_ny = 14;
  geom_nz = 3;
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
            {"name", "SetGradientField"},
            {"value", {0.0, 0.0, B0_value}},
            {"axis", "X"},
            {"length", L_value},
            // Shift the origin by half a cell to compensate the Yee stagger of
            // Bz, so the field the particle samples at r0[X] is exactly B0_value.
            {"origin", r0[X] - 0.5 * cell},
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
