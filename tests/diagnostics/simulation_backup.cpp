#include "src/diagnostics/simulation_backup.h"

#include "src/interfaces/particles.h"
#include "src/utils/geometries.h"
#include "src/utils/world.h"
#include "tests/common.h"

static char help[] = "Testing save and load of simulation backup\n.";

bool operator==(const Point& a, const Point& b);

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  World::set_geometry(REP3(10.0), 1.0, REP3(1.0), 1.0, 1.0);

  World world;
  PetscCall(world.initialize());

  Vec v;
  PetscCall(DMCreateGlobalVector(world.da, &v));

  PetscRandom rnd;
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rnd));
  PetscCall(PetscRandomSetType(rnd, PETSCRAND48));
  PetscCall(PetscRandomSetSeed(rnd, 0));
  PetscCall(VecSetRandom(v, rnd));
  PetscCall(PetscRandomDestroy(&rnd));

  static const Vector3R p0{0.0};
  static const std::vector<Point> prepared_points{
    Point{Vector3R{2, 2, 2}, p0},
    Point{Vector3R{8, 2, 2}, p0},
    Point{Vector3R{2, 8, 2}, p0},
    Point{Vector3R{8, 8, 2}, p0},
    Point{Vector3R{2, 2, 8}, p0},
    Point{Vector3R{8, 2, 8}, p0},
    Point{Vector3R{2, 8, 8}, p0},
    Point{Vector3R{8, 8, 8}, p0},
  };

  using Particles = interfaces::Particles;
  auto particles = std::make_unique<Particles>(world, SortParameters{});

  for (const auto& point : prepared_points)
    PetscCall(particles->add_particle(point));

  std::filesystem::path out_dir = get_out_dir(__FILE__);

  /// @note We should create the diagnostic within some scope to properly run the destructors.
  {
    std::map<std::string, Vec> _fields{{"field", v}};
    std::map<std::string, Particles*> _particles{{"particles", particles.get()}};

    auto&& diag =
      std::make_unique<SimulationBackup>(out_dir, 1, _fields, _particles);

    // Backup of constructed example, it is written to `./output/simulation_backup/`
    PetscCall(diag->save(0));

    uintmax_t size = std::filesystem::file_size(out_dir.string() + "/0/field");
    uintmax_t csize = sizeof(PetscInt) * 2 + sizeof(PetscReal) * 3 * POW3(10);
    PetscCheck(size == csize, PETSC_COMM_WORLD, PETSC_ERR_USER, "Saved fields data is inconsistent");

    size = std::filesystem::file_size(out_dir.string() + "/0/particles");
    csize = sizeof(PetscReal) * 6 * prepared_points.size();
    PetscCheck(size == csize, PETSC_COMM_WORLD, PETSC_ERR_USER, "Saved particles data is inconsistent");


    // Clearing up field and particles to load the backup
    Vec tmp;
    PetscCall(VecDuplicate(v, &tmp));
    PetscCall(VecSwap(v, tmp));

    for (auto& cell : particles->storage)
      cell.clear();

    // Loading the backup into "field" `v` and "particles" `particles`
    PetscCall(diag->load(0));

    PetscReal norm;
    PetscCall(VecAXPY(tmp, -1, v));
    PetscCall(VecNorm(tmp, NORM_2, &norm));
    PetscCheck(norm < PETSC_MACHINE_EPSILON, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Field data is loaded incorrectly, norm2 error: %f", norm);

    for (const auto& point : prepared_points) {
      Vector3I vg{
        FLOOR_STEP(point.x(), dx) - world.start[X],
        FLOOR_STEP(point.y(), dy) - world.start[Y],
        FLOOR_STEP(point.z(), dz) - world.start[Z],
      };

      if (!is_point_within_bounds(vg, 0, world.size))
        continue;

      PetscInt g = world.s_g(REP3_A(vg));
      const auto& cell = particles->storage[g];

      PetscCheck(cell.size() == 1, PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Particles data is loaded incorrectly, cell.size() is %" PetscInt_FMT " at g: %" PetscInt_FMT, (PetscInt)cell.size(), g);

      PetscCheck(*cell.begin() == point, PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Particles data is incorrect, wrong particle point at g: %" PetscInt_FMT, g);
    }

    PetscCall(VecDestroy(&tmp));
  }

  PetscCall(VecDestroy(&v));
  PetscCall(world.finalize());

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}


bool operator==(const Point& a, const Point& b)
{
  return  //
    std::abs(a.r[X] - b.r[X]) < PETSC_MACHINE_EPSILON &&
    std::abs(a.r[Y] - b.r[Y]) < PETSC_MACHINE_EPSILON &&
    std::abs(a.r[Z] - b.r[Z]) < PETSC_MACHINE_EPSILON &&
    std::abs(a.p[X] - b.p[X]) < PETSC_MACHINE_EPSILON &&
    std::abs(a.p[Y] - b.p[Y]) < PETSC_MACHINE_EPSILON &&
    std::abs(a.p[Z] - b.p[Z]) < PETSC_MACHINE_EPSILON;
}
