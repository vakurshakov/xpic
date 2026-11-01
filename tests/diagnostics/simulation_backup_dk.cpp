#include "src/diagnostics/simulation_backup.h"

#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing save and load of simulation backup for drift-kinetic particles.\n"
  "All arrays of the storage but `_prev` ones are filled with distinct    \n"
  "values, saved, cleared and loaded back.                                \n";

static constexpr PetscReal q = -1.0;
static constexpr PetscReal m = 2.0;
static constexpr PetscReal n = 1.0;
static constexpr PetscInt Np = 3;
static constexpr PetscReal B_value = 2.0;

const std::vector<Point> points{
  Point{
    .r = Vector3R{2, 3, 4},
    .p = Vector3R{0.1, 0.2, 0.3},
  },
  Point{
    .r = Vector3R{5.5, 5, 5},
    .p = Vector3R{-0.3, 0.0, 0.4},
  },
  Point{
    .r = Vector3R{7.5, 0.5, 6.5},
    .p = Vector3R{0.2, -0.1, -0.4},
  },
};

void overwrite_config();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();

  drift_kinetic::Simulation simulation;
  PetscCall(simulation.initialize());

  auto& particles = *simulation.particles_[0];
  auto& storage = particles.storage;

  // Making every non-position array distinct, so that no field can be mixed up with another
  ParticleStorage::Fields f;
  PetscCall(storage.get_fields(f));
  PetscInt k = 1;
  for (const auto& [name, dof] : f.dof) {
    k++;
    if (name == "r")
      continue;
    for (PetscInt i = 0; i < f.size; i++) {
      for (PetscInt c = 0; c < dof; c++)
        f[name][dof * i + c] =        //
          k + 0.01 * c +              //
          0.1 * f["r"][3 * i + X] +   //
          0.001 * f["r"][3 * i + Y];  //
    }
  }
  PetscCall(storage.restore_fields(f));

  Snapshot before;
  PetscCall(take_snapshot(storage, before));

  PetscInt local = (PetscInt)before.size(), total = 0;
  PetscCallMPI(MPI_Allreduce(&local, &total, 1, MPIU_INT, MPIU_SUM, PETSC_COMM_WORLD));
  PetscCheck(total == Np * (PetscInt)points.size(), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Unexpected number of loaded particles: %" PetscInt_FMT, total);

  std::vector<interfaces::Particles*> sorts{&particles};
  SimulationBackup backup(get_out_dir(__FILE__) / "backup", 1, {}, sorts);

  PetscCall(backup.save(0));
  PetscCall(storage.clear());
  PetscCall(backup.load(0));

  Snapshot after;
  PetscCall(take_snapshot(storage, after));
  PetscCheck(after == before, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Drift-kinetic storage differs after the backup is loaded");

  PetscCall(simulation.finalize());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config();

PetscErrorCode check_moment(const drift_kinetic::Particles& particles,
  const std::string& name, PetscInt dof);

void overwrite_config()
{
  dx = dy = dz = 1.0;
  geom_nx = geom_ny = geom_nz = 8;
  geom_x = geom_y = geom_z = 8.0;

  dt = 1.0;
  geom_nt = 1;
  geom_t = 1.0;

  const char* sort_name = "particles";

  Configuration::array_t presets;

  presets.push_back({
    {"command", "SetField"},
    {"field", "B"},
    {"setter",
      {
        {"name", "SetUniformField"},
        {"value", {0.0, 0.0, B_value}},
      }},
  });

  for (const Point& point : points) {
    presets.push_back({
      // `PreciseCoordinate` loads `Np` copies
      {"command", "SetParticles"},
      {"particles", sort_name},
      {"coordinate",
        {
          {"name", "PreciseCoordinate"},
          {"value", {point.r[X], point.r[Y], point.r[Z]}},
        }},
      {"momentum",
        {
          {"name", "PreciseMomentum"},
          {"value", {point.p[X], point.p[Y], point.p[Z]}},
        }},
    });
  }

  Configuration::overwrite({
    {"Simulation", "drift_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {"DriftKineticPush", {{"coord_is_gc", true}}},
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
      {{
        {"sort_name", sort_name},
        {"Np", Np},
        {"n", n},
        {"q", q},
        {"m", m},
      }},
    },
    {"Presets", presets},
  });
}
