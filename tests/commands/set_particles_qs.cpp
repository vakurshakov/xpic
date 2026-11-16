#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/random_generator.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing QS initialization: particle counts, paired coordinates, initial\n"
  "charge neutrality and density noise. No time evolution is performed.\n";

namespace {
constexpr PetscInt particles_per_cell = 32;

PetscErrorCode run_case(
  const Configuration::json_t& coordinate, PetscReal& density_rms)
{
  PetscFunctionBeginUser;
  Configuration::json_t presets = {
    {{"command", "SetField"}, {"field", "B"},
      {"setter", {{"name", "SetUniformField"}, {"value", {0, 0, 1}}}}},
  };
  presets.push_back({
    {"command", "SetParticles"},
    {"particles", "ions"}, {"paired_with", "electrons"},
    {"coordinate", coordinate},
    {"momentum", {{"name", "MaxwellianMomentum"}}},
    {"momentum_paired", {{"name", "MaxwellianMomentum"}}},
  });
  Configuration::overwrite({
    {"Simulation", "drift_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {"DriftKineticPush", {{"coord_is_gc", true}}},
    {"FieldPreconditioner", "none"},
    {"Geometry",
      {
        {"x", 80}, {"y", 80}, {"z", 80}, {"t", 10},
        {"dx", 10}, {"dy", 10}, {"dz", 10}, {"dt", 10},
        {"diagnose_period", 10},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      }},
    {"Particles",
      {
        {{"sort_name", "ions"}, {"model", "drift_kinetic"},
          {"Np", particles_per_cell}, {"n", 1},
          {"q", 1}, {"m", 100}, {"T", 0.1}},
        {{"sort_name", "electrons"}, {"model", "drift_kinetic"},
          {"Np", particles_per_cell}, {"n", 1},
          {"q", -1}, {"m", 1}, {"T", 0.1}},
      }},
    {"Presets", presets},
  });
  RandomGenerator::get().seed(1729);
  drift_kinetic::Simulation simulation;
  PetscCall(simulation.initialize());
  auto& electrons = simulation.get_named_particles("electrons");
  auto& ions = simulation.get_named_particles("ions");
  PetscReal charge_norm;
  Vec charge;
  PetscCall(VecDuplicate(electrons.rho[1], &charge));
  PetscCall(VecWAXPY(charge, 1, ions.rho[1], electrons.rho[1]));
  PetscCall(VecNorm(charge, NORM_INFINITY, &charge_norm));
  PetscCall(VecDestroy(&charge));

  PetscCheck(charge_norm < 1e-12, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Paired loading must be neutral, charge norm %g", charge_norm);

  PetscInt expected = geom_nx * geom_ny * geom_nz * particles_per_cell;
  if (coordinate.at("name") == "CoordinateInCylinderQS")
    expected =
      std::round(M_PI * 100 * 80 / (dx * dy * dz) * particles_per_cell);

  std::array<std::vector<std::array<PetscReal, 3>>, 2> coordinates;
  std::size_t s = 0;
  for (auto* particles : {&ions, &electrons}) {
    PetscInt count;
    PetscCall(particles->storage.get_size(&count));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

    PetscCheck(count == expected, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Loaded %" PetscInt_FMT " particles, expected %" PetscInt_FMT, count, expected);

    ParticleStorage::Attributes f;
    PetscCall(particles->storage.get_attribute(f, "r"));
    for (PetscInt i = 0; i < f.size; i++) {
      coordinates[s].push_back({
        f["r"][3 * i], f["r"][3 * i + 1], f["r"][3 * i + 2],
      });
    }
    PetscCall(particles->storage.restore_attributes(f));
    std::ranges::sort(coordinates[s]);
    s++;
  }

  PetscCheck(coordinates[0] == coordinates[1], PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Paired sorts must share particle coordinates on every rank");

  PetscCall(VecShift(electrons.rho[1], 1));
  PetscCall(VecNorm(electrons.rho[1], NORM_2, &density_rms));
  density_rms /= std::sqrt(geom_nx * geom_ny * geom_nz);
  PetscCall(simulation.finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode run()
{
  PetscFunctionBeginUser;
  PetscReal halton_density, random_density;
  PetscCall(run_case({{"name", "CoordinateInBoxQS"}}, halton_density));
  PetscCall(run_case({{"name", "CoordinateInBox"}}, random_density));

  PetscCheck(halton_density < random_density, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Initial Halton density noise %g must be below random %g", halton_density, random_density);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "Density RMS: Halton %g, random %g\n", halton_density, random_density));

  PetscReal density;
  PetscCall(run_case({{"name", "CoordinateInCylinderQS"},
    {"center", {40, 40, 40}}, {"radius", 10}, {"height", 80}}, density));
  for (const char* axis : {"X", "Y", "Z"})
    PetscCall(run_case({{"name", "CoordinateInBoxCosineHumpQS"}, {"axis", axis}}, density));

  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(run());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
