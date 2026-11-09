#include "src/interfaces/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing \'LogView\' diagnostics of all levels for the simulation given by \n"
  "`-simulation`. Only the creation of non-empty files is checked, since the \n"
  "timings are not reproducible.\n";

void overwrite_config(const char* name);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  char name[PETSC_MAX_PATH_LEN] = "";
  PetscCall(PetscOptionsGetString(NULL, NULL, "-simulation", name, sizeof(name), NULL));

  // The tables of other simulations from the shared directory must not be compared
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0) {
    std::filesystem::remove_all(get_out_dir(__FILE__));
  }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  overwrite_config(name);

  auto simulation = build_simulation();
  PetscCall(simulation->initialize());
  PetscCall(simulation->calculate());
  PetscCall(simulation->finalize());

  const char* levels[]{
    "EachTimestep",
    "DiagnosePeriodAvg",
    "AllTimestepsSummary",
    "EachTimestepEvents",
  };

  for (const char* level : levels) {
    auto path = get_out_dir(__FILE__) / std::format("log-{}.txt", level);
    PetscCheck(std::filesystem::exists(path) && std::filesystem::file_size(path) > 0, PETSC_COMM_SELF, PETSC_ERR_USER,
      "\"%s\" must be written and non-empty", path.c_str());
  }

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config(const char* name)
{
  dx = dy = dz = 1.0;
  geom_nx = geom_ny = geom_nz = 8;
  geom_x = geom_y = geom_z = geom_nx * dx;

  dt = 1.0;
  geom_nt = 1;
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", std::string(name)},
    {"OutputDirectory", get_out_dir(__FILE__)},
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
      {{
        {"command", "SetParticles"},
        {"particles", "electrons"},
        {"coordinate", {{"name", "CoordinateInBox"}}},
        {"momentum", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
      }},
    },
    {
      "Diagnostics",
      {
        {{"diagnostic", "LogView"}, {"level", "EachTimestep"}},
        {{"diagnostic", "LogView"}, {"level", "DiagnosePeriodAvg"}},
        {{"diagnostic", "LogView"}, {"level", "AllTimestepsSummary"}},
        {{"diagnostic", "LogView"}, {"level", "EachTimestepEvents"}},
      },
    },
  });
}
