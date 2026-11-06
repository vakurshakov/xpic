#ifndef TESTS_HYBRID_HYBRID_EX1_COMPARISON_H
#define TESTS_HYBRID_HYBRID_EX1_COMPARISON_H

#include "src/utils/configuration.h"
#include "src/utils/utils.h"
#include "tests/common.h"

/// @brief Shared setup of the energy and charge conservation tests: a uniform
/// Maxwellian electron-ion plasma in a uniform magnetic field, 3x3x3 cells
/// with dx = 10 >> r_D and periodic boundaries (section 4.2 of the paper).
/// The defaults are overridden by the options `-dx`, `-tau` (in 1/Omega_e),
/// `-time`, `-npc`, `-mi`, `-precond` ("FieldPreconditioner") and `-tag`
/// to scan the parameters.
namespace ex1 {

constexpr PetscReal cell = 10.0;
constexpr PetscInt cells = 3;
constexpr PetscReal B_value = 1.0;  // Omega_e = 1, so dt is in 1/Omega_e.
constexpr PetscReal total_time = 10000.0;
constexpr PetscInt particles_per_cell = 50;
constexpr PetscReal ion_mass = 100.0;
constexpr PetscReal temperature = 0.1;            // [keV], both sorts.
constexpr const char* field_tolerance = "1e-13";  // SNES atol and rtol.

/// @brief Output directory, suffixed with `_<tag>` if `-tag` is set.
std::filesystem::path out_dir(std::string_view file)
{
  char tag[PETSC_MAX_PATH_LEN] = "";
  PetscCallAbort(PETSC_COMM_WORLD, PetscOptionsGetString(nullptr, nullptr, "-tag", tag, sizeof(tag), nullptr));

  std::filesystem::path result = get_out_dir(file);
  if (tag[0] == '\0')
    return result;
  return std::format("{}_{}/", result.parent_path().c_str(), tag);
}

/// @brief `ions_model` and `electrons_model` are used by `hybrid_kinetic` only.
PetscErrorCode overwrite_config(std::string_view file,
  const std::string& simulation, const std::string& ions_model,
  const std::string& electrons_model, PetscReal step)
{
  PetscFunctionBeginUser;
  PetscReal time = total_time;
  PetscReal mi = ion_mass;
  PetscInt npc = particles_per_cell;
  char precond[PETSC_MAX_PATH_LEN] = "none";
  dx = cell;
  dt = step;

  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-dx", &dx, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-tau", &dt, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-time", &time, nullptr));
  PetscCall(PetscOptionsGetReal(nullptr, nullptr, "-mi", &mi, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-npc", &npc, nullptr));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-precond", precond, sizeof(precond), nullptr));

  dy = dz = dx;
  geom_nx = geom_ny = geom_nz = cells;
  geom_x = geom_nx * dx;
  geom_y = geom_ny * dy;
  geom_z = geom_nz * dz;

  geom_nt = ROUND_STEP(time, dt);
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", simulation},
    {"OutputDirectory", out_dir(file)},
    {"DriftKineticPush", {{"coord_is_gc", true}}},
    {"FieldPreconditioner", precond},
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
      {
        {
          {"sort_name", "ions"},
          {"model", ions_model},
          {"Np", npc},
          {"n", +1.0},
          {"q", +1.0},
          {"m", mi},
          {"T", temperature},
        },
        {
          {"sort_name", "electrons"},
          {"model", electrons_model},
          {"Np", npc},
          {"n", +1.0},
          {"q", -1.0},
          {"m", +1.0},
          {"T", temperature},
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
        // Guiding centers are placed exactly at the ion positions,
        // so the initial charge density is zero and div(E) = rho holds.
        {
          {"command", "SetParticles"},
          {"particles", "ions"},
          {"paired_with", "electrons"},
          {"coordinate", {{"name", "CoordinateInBox"}}},
          {"momentum", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
          {"momentum_paired", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
        },
      },
    },
  });
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Runs the simulation with the shared field solver tolerances (unless
/// set by the options) and writes the wall time of its phases to
/// `temporal/timing.txt`.
template<class Simulation>
PetscErrorCode run_timed(std::string_view file)
{
  PetscFunctionBeginUser;
  for (const char* name : {"-snes_atol", "-snes_rtol"}) {
    PetscBool is_set;
    PetscCall(PetscOptionsHasName(nullptr, nullptr, name, &is_set));
    if (!is_set)
      PetscCall(PetscOptionsSetValue(nullptr, name, field_tolerance));
  }
  PetscCall(PetscOptionsSetValue(nullptr, "-snes_stol", "0"));

  Simulation simulation;
  PetscReal time[4];

  time[0] = MPI_Wtime();
  PetscCall(simulation.initialize());
  time[1] = MPI_Wtime();
  PetscCall(simulation.calculate());
  time[2] = MPI_Wtime();
  PetscCall(simulation.finalize());
  time[3] = MPI_Wtime();

  const PetscReal init = time[1] - time[0];
  const PetscReal calc = time[2] - time[1];
  const PetscReal fin = time[3] - time[2];

  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  if (rank == 0) {
    std::ofstream timing(out_dir(file) / "temporal" / "timing.txt");
    timing << std::format("{:<13}  {:<13}  {:<13}  {:<13}  {:<6}  {:<13}\n",
      "Initialize", "Calculate", "Finalize", "Total", "Steps", "PerStep");
    timing << std::format(
      "{:.6e}   {:.6e}   {:.6e}   {:.6e}   {:<6d}  {:.6e}\n",  //
      init, calc, fin, time[3] - time[0], geom_nt, calc / geom_nt);
  }

  LOG("Wall time: initialize {:.3f} s, calculate {:.3f} s ({:.3e} s per step), finalize {:.3f} s",
    init, calc, calc / geom_nt, fin);
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace ex1

#endif  // TESTS_HYBRID_HYBRID_EX1_COMPARISON_H
