#include "src/interfaces/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing SetParticlesIonAcousticWave: every particle pair must share \n"
  "its coordinate and have mirrored thermal velocities.                \n";

static constexpr PetscInt Np = 4;
static constexpr PetscReal volume = 4.0 * 4.0 * 8.0;

using json_t = Configuration::json_t;

PetscErrorCode run();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(run());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

json_t ion_acoustic_wave(const std::string& sort)
{
  return {
    {"command", "SetParticlesIonAcousticWave"},
    {"particles", sort},
    {"wave_number", {0.0, 0.0, 1.0}},
    {"density_amplitude", {0.0, 0.0, 0.01}},
    {"force_electric_amplitude", 1e-8},
    {"omega_real", 0.01},
    {"gamma", 0.01},
  };
}

void overwrite_config()
{
  dx = dy = dz = 1.0;
  geom_nx = geom_ny = 4;
  geom_nz = 8;
  geom_x = geom_y = 4.0;
  geom_z = 8.0;

  dt = 0.1;
  geom_nt = 1;
  geom_t = dt;

  Configuration::overwrite({
    {"Simulation", "basic"},
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
    {"Fields", {{"evolve", false}}},
    {
      "Particles",
      {
        {
          {"sort_name", "a"},
          {"Np", Np},
          {"n", +1.0},
          {"q", -1.0},
          {"m", +1.0},
          {"T", +0.1},
        },
        {
          {"sort_name", "b"},
          {"Np", Np},
          {"n", +1.0},
          {"q", +1.0},
          {"m", +4.0},
          {"T", +0.1},
        },
      },
    },
    {
      "Presets",
      {
        ion_acoustic_wave("a"),
        ion_acoustic_wave("b"),
      },
    },
  });
}

PetscErrorCode check_pairs(const interfaces::Particles& particles)
{
  PetscFunctionBeginUser;
  std::map<std::array<PetscReal, 3>, PetscInt> multiplicity;
  PetscInt count = 0;
  PetscReal sum_px = 0;
  PetscReal sum_py = 0;

  for (const auto& cell : particles.storage) {
    for (const Point& point : cell) {
      multiplicity[{point.x(), point.y(), point.z()}]++;
      sum_px += point.px();
      sum_py += point.py();
      count++;
    }
  }

  // Pairs are never split between ranks: they share the coordinate
  for (const auto& [coordinate, number] : multiplicity)
    PetscCheck(number == 2, PETSC_COMM_WORLD, PETSC_ERR_USER, "A coordinate is shared by %d particles, expected 2", number);

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &sum_px, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &sum_py, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));

  const char* name = particles.parameters.sort_name.c_str();
  const PetscInt expected = (PetscInt)(volume * 2 * Np);

  PetscCheck(count == expected, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "\"%s\": %d particles, expected %d", name, count, expected);

  // Perpendicular velocity has no bulk part, mirrored thermal parts cancel
  PetscCheck(std::abs(sum_px) < 1e-12 && std::abs(sum_py) < 1e-12, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "\"%s\": perpendicular velocity sum (%g, %g) is not zero", name, sum_px, sum_py);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode run()
{
  PetscFunctionBeginUser;
  overwrite_config();

  std::unique_ptr<interfaces::Simulation> simulation = build_simulation();
  PetscCall(simulation->initialize());

  for (const char* name : {"a", "b"})
    PetscCall(check_pairs(simulation->get_named_particles(name)));

  PetscCall(simulation->finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
