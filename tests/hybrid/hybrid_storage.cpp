#include "src/impls/hybrid_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

namespace {

constexpr PetscReal positions[] = {3.9, 5.9, 7.9, 11.9};
constexpr PetscReal tolerance = 1e-9;

PetscErrorCode check(const interfaces::Particles& sort)
{
  PetscFunctionBeginUser;
  ParticleStorage::Attributes f;
  PetscCall(sort.storage.get_attributes(f));
  const PointArrays arrays(f);
  PetscInt counts[4] = {};
  for (PetscInt i = 0; i < f.size; i++) {
    const Point point = arrays.load(i);
    const PetscInt id = (PetscInt)std::round(10 * point.p[X]) - 2;
    PetscCheck(id >= 0 && id < 4, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
      "Particle velocity was corrupted");
    const PetscReal v = 0.2 + 0.1 * id;
    const PetscReal x = std::fmod(positions[id] + geom_nt * dt * v, geom_x);
    PetscCheck((point.r - Vector3R{x, 1.5, 1.5}).abs_max() < tolerance &&
        (point.p - Vector3R{v, 0, 0}).abs_max() < tolerance,
      PETSC_COMM_WORLD, PETSC_ERR_PLIB,
      "Current and previous particle states lost correspondence");
    if (f.data.contains("mu_p")) {
      PetscCheck(std::abs(f["p_parallel"][i] - v) < tolerance &&
          std::abs(f["mu_p"][i]) < tolerance,
        PETSC_COMM_WORLD, PETSC_ERR_PLIB, "DK attributes were corrupted");
    }
    counts[id]++;
  }
  PetscCall(sort.storage.restore_attributes(f));
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, counts, 4, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  for (PetscInt count : counts) {
    PetscCheck(count == 1, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
      "A particle was lost or duplicated during migration");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, nullptr));
  PetscBool frozen = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr, "-hybrid_frozen", &frozen, nullptr));

  World::set_geometry(12.0, 12.0, 12.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0);
  Configuration::array_t presets{
    {{"command", "SetField"}, {"field", "B"},
      {"setter", {{"name", "SetUniformField"}, {"value", {1, 0, 0}}}}},
  };
  Configuration::array_t sorts;
  for (const std::string model : {"drift_kinetic", "kinetic"}) {
    sorts.push_back({{"sort_name", model}, {"model", model}, {"Np", 1},
      {"n", 1}, {"q", model == "kinetic" ? 1 : -1}, {"m", 1}});
    for (PetscInt i = 0; i < 4; i++) {
      presets.push_back({
        {"command", "SetParticles"}, {"particles", model},
        {"coordinate", {{"name", "PreciseCoordinate"},
          {"value", {positions[i], 1.5, 1.5}}}},
        {"momentum", {{"name", "PreciseMomentum"},
          {"value", {0.2 + 0.1 * i, 0, 0}}}},
      });
    }
  }
  Configuration::overwrite({
    {"Simulation", "hybrid_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {"Fields", {{"evolve", !frozen}}},
    {"DriftKineticPush", {{"coord_is_gc", true}}},
    {"Geometry", {
      {"x", 12}, {"y", 12}, {"z", 12}, {"t", 2},
      {"dx", 1}, {"dy", 1}, {"dz", 1}, {"dt", 1},
      {"diagnose_period", 1},
      {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
      {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
      {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
    }},
    {"Particles", sorts},
    {"Presets", presets},
  });
  Configuration::save(get_out_dir(__FILE__));
  hybrid_kinetic::Simulation simulation;
  PetscCall(simulation.initialize());
  PetscCall(simulation.calculate());
  PetscCall(check(*simulation.particles_.at(0)));
  PetscCall(check(*simulation.kinetic_particles_.at(0)));
  PetscReal norm;
  PetscCall(VecNorm(simulation.E, NORM_INFINITY, &norm));
  PetscCheck(norm < tolerance, PETSC_COMM_WORLD, PETSC_ERR_PLIB,
    "Opposite currents did not cancel");
  PetscCall(simulation.finalize());
  PetscCall(PetscFinalize());
  return 0;
}
