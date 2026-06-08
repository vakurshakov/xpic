#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Ion-sound wave test for \"drift_kinetic\" implementation.                \n"
  "Plasma in a thin (5*dx x 5*dx) periodic column of length Lz = 200 c/wpe \n"
  "is given a small density perturbation n = 1 + 0.03 sin(2 pi z/Lz)       \n"
  "and a phase-locked bulk velocity V_z = 0.03 sqrt(Te/me c^2) sin(2 pi z/Lz),\n"
  "with electrons at Te = 20 keV and ions of mass 64 m_e at Ti = 0.01.     \n";

void overwrite_config();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();
  Configuration::save(get_out_dir(__FILE__));

  drift_kinetic::Simulation simulation;
  PetscCall(simulation.initialize());
  PetscCall(simulation.calculate());
  PetscCall(simulation.finalize());

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config()
{
  dx = 10;
  geom_nx = 3;
  geom_x = geom_nx * dx;
  geom_ny = 3;
  geom_y = geom_ny * dx;
  geom_nz = 20;
  geom_z = geom_nz * dx;  // Lz = 200 c/wpe

  dt = 10;
  geom_nt = 6000;
  geom_t = geom_nt * dt;

  // V_amp = 0.03 * sqrt(Te / (me c^2)) = 0.03 * sqrt(20 / 511).
  const double v_amp = 0.;//0.03 * std::sqrt(20.0 / 64/ 511.0);

  Configuration::overwrite({
    {"Simulation", "drift_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {
      "Geometry",
      {
        {"x", geom_x},
        {"y", geom_y},
        {"z", geom_z},
        {"t", geom_t},
        {"dx", dx},
        {"dy", dx},
        {"dz", dx},
        {"dt", dt},
        {"diagnose_period", 10*dt},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "electrons"},
        {"Np", 10000},
        {"n", +1.0},
        {"q", -1.0},
        {"m", +1.0},
        {"T", +20.0},
        // Treat sampled `r` as the guiding center (no Larmor shift on load).
        // Combined with the paired loader below, this places electron and ion
        // GCs at exactly the same position for every macro-particle pair.
        {"coord_is_gc", true},
      },
      {
        {"sort_name", "ions"},
        {"Np", 10000},
        {"n", +1.0},
        {"q", +1.0},
        {"m", +64.0},
        {"T", +0.01},
        {"coord_is_gc", true},
      }},
    },
    {
      "Presets",
      {
        {
          {"command", "SetMagneticField"},
          {"field", "B0"},
          {"field_axpy", "B"},
          {
            "setter",
            {
              {"name", "SetUniformField"},
              {"value", {0.0, 0.0, 1.0}},
            },
          },
        },
        // Paired loader: one shared `r` per macro-particle pair (no RNG drift
        // between species). With `coord_is_gc=true` on both sorts, the same
        // `r` becomes the guiding center of both the electron and the ion —
        // i.e. ||r_e_gc - r_i_gc|| == 0 for every pair.
        //
        // - `particles`     — first sort of the pair (loaded as `ionized` slot).
        // - `paired_with`   — second sort name (loaded as `ejected` slot).
        // - `coordinate`    — shared coordinate generator.
        // - `momentum`      — momentum generator for `particles`.
        // - `momentum_paired` — momentum generator for `paired_with`.
        {
          {"command", "SetParticles"},
          {"particles", "electrons"},
          {"paired_with", "ions"},
          {"coordinate", {
            // Quiet-start loader: Halton low-discrepancy sequence along the
            // perturbed axis (here z) + analytic O(A) displacement that
            // reproduces n(z) = 1 + amplitude_z * sin(2 pi k_z z / Lz).
            // Removes the residual ~1/sqrt(N) Poisson noise of the plain
            // displacement loader — the only remaining error is the
            // O(amplitude^2) systematic of the linear-shift formula.
            {"name", "CoordinateInBoxQuietSine"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"amplitude", {0.0, 0.0, 0.03}},
            {"wave_number", {0.0, 0.0, 1.0}},
          }},
          {"momentum", {
            {"name", "MaxwellShiftedSine"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"velocity", {0.0, 0.0, v_amp}},
            {"wave_number", {0.0, 0.0, 1.0}},
          }},
          {"momentum_paired", {
            {"name", "MaxwellShiftedSine"},
            {"min", {0.0, 0.0, 0.0}},
            {"max", {geom_x, geom_y, geom_z}},
            {"velocity", {0.0, 0.0, v_amp}},
            {"wave_number", {0.0, 0.0, 1.0}},
          }},
        },
      },
    },
    {
      "Diagnostics",
      {
        {
          {"diagnostic", "FieldView"},
          {"field", "E"},
          {"out_dir", "E_planeY"},
          {"region", {{"type", "2D"}, {"plane", "Y"}}},
        },
        {
          {"diagnostic", "FieldView"},
          {"field", "B"},
          {"out_dir", "B_planeY"},
          {"region", {{"type", "2D"}, {"plane", "Y"}}},
        },
        {
          {"diagnostic", "FieldView"},
          {"field", "electrons/J"},
          {"out_dir", "electrons/J"},
        },
        {
          {"diagnostic", "FieldView"},
          {"field", "ions/J"},
          {"out_dir", "ions/J"},
        },
        {
          {"diagnostic", "DistributionMoment"},
          {"particles", "electrons"},
          {"moment", "density"},
        },
        {
          {"diagnostic", "DistributionMoment"},
          {"particles", "ions"},
          {"moment", "density"},
        },
      },
    },
  });
}
