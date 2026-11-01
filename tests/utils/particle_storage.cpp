#include "src/utils/particle_storage.h"

#include "tests/common.h"

static constexpr char help[] =
  "Testing ParticleStorage: the previous attributes, the exchange of particles\n"
  "between ranks and the counting sort by cell.\n";

constexpr PetscInt ppc = 8;
constexpr PetscInt steps = 3;
constexpr PetscReal tol = 1e-10;

// Particle `id` is identified by `p = (id, 2 id, -id)`, which travels with it
static PetscReal initial_position(PetscInt id, PetscInt axis)
{
  static constexpr PetscReal golden[3] = {
    0.6180339887,
    0.4142135623,
    0.7320508075,
  };

  const PetscInt k = id % ppc;
  PetscInt cell = id / ppc;
  for (PetscInt a = 0; a < axis; a++)
    cell /= Geom_n[a];

  return  //
    (cell % Geom_n[axis] + std::fmod((k + 1) * golden[axis], 1.0)) * Dx[axis];
}

// Up to 1.4 cells per step, so that particles also cross the edges and corners
static PetscReal displacement(PetscInt id, PetscInt axis)
{
  return ((id * 7 + 3 * axis) % 29 - 14) * 0.1 * Dx[axis];
}

static PetscErrorCode check_particles(ParticleStorage& storage, PetscInt step)
{
  PetscFunctionBeginUser;
  ParticleStorage::Attributes f;
  PetscCall(storage.get_attributes(f));

  PetscReal sum = 0;
  for (PetscInt i = 0; i < f.size; i++) {
    const PetscReal* r = &f["r"][3 * i];
    const PetscReal* p = &f["p"][3 * i];
    const PetscInt id = (PetscInt)std::lround(p[X]);

    PetscCheck(p[Y] == 2.0 * id && p[Z] == -id, PETSC_COMM_WORLD, PETSC_ERR_USER, "Momentum of particle %d is corrupted", id);

    for (PetscInt a = 0; a < 3; a++) {
      PetscReal x = initial_position(id, a) + step * displacement(id, a);
      PetscReal d = std::fmod(std::abs(r[a] - x), Geom[a]);
      d = std::min(d, Geom[a] - d);
      PetscCheck(d < tol, PETSC_COMM_WORLD, PETSC_ERR_USER, "Particle %d is misplaced along axis %d: %g, expected %g mod %g", id, a, r[a], x, Geom[a]);
    }
    sum += id;
  }

  PetscInt n = f.size;
  PetscCall(storage.restore_attributes(f));

  PetscReal total_sum;
  PetscInt total;
  PetscCallMPI(MPI_Allreduce(&sum, &total_sum, 1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&n, &total, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

  const PetscInt expected = ppc * Geom_n[X] * Geom_n[Y] * Geom_n[Z];
  PetscCheck(total == expected && total_sum == 0.5 * expected * (expected - 1), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Particles are lost or duplicated: %d of %d, sum of ids %g", total, expected, total_sum);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  // dx != dy != dz to catch axis mix-ups
  World::set_geometry(8.0, 5.0, 4.0, 1.0, 1.0, 0.5, 0.25, 1.0, 1.0);

  World world;
  for (DMBoundaryType& bound : world.bounds)
    bound = DM_BOUNDARY_PERIODIC;
  PetscCall(world.initialize());

  ParticleStorage storage(world);
  PetscCall(storage.initialize());
  PetscCall(storage.register_attribute("r", 3));
  PetscCall(storage.register_attribute("p", 3));
  PetscCall(storage.register_previous("r"));
  PetscCall(storage.register_previous("p"));
  PetscCall(storage.commit_attributes());

  PetscInt first;
  PetscCall(storage.grow(ppc * world.size.elements_product(), &first));

  ParticleStorage::Attributes f;
  PetscCall(storage.get_attributes(f));

  PetscInt i = 0;
  for (PetscInt z = world.start[Z]; z < world.end[Z]; z++) {
    for (PetscInt y = world.start[Y]; y < world.end[Y]; y++) {
      for (PetscInt x = world.start[X]; x < world.end[X]; x++) {
        for (PetscInt p = 0; p < ppc; p++, i++) {
          const PetscInt id = ((z * Geom_n[Y] + y) * Geom_n[X] + x) * ppc + p;
          for (PetscInt a = 0; a < 3; a++)
            f["r"][3 * i + a] = initial_position(id, a);
          f["p"][3 * i + X] = id;
          f["p"][3 * i + Y] = 2.0 * id;
          f["p"][3 * i + Z] = -id;
        }
      }
    }
  }
  PetscCall(storage.restore_attributes(f));
  PetscCall(check_particles(storage, 0));

  // Getters load only what is asked for
  PetscCall(storage.get_attributes(f));
  PetscCheck(f.data.size() == 2 && !f.data.contains("p_prev"), PETSC_COMM_WORLD, PETSC_ERR_USER, "`get_attributes()` must not load the previous arrays");
  PetscCall(storage.restore_attributes(f));

  PetscCall(storage.get_attribute(f, "p_prev"));
  PetscCheck(f.data.size() == 1 && f.dof.at("p_prev") == 3, PETSC_COMM_WORLD, PETSC_ERR_USER, "`get_attribute()` must load exactly one array");
  PetscCall(storage.restore_attributes(f));

  for (PetscInt step = 1; step <= steps; step++) {
    PetscCall(storage.save_previous());
    PetscCall(storage.get_attributes(f));
    PetscCall(storage.get_previous(f));
    for (PetscInt i = 0; i < f.size; i++) {
      const PetscInt id = (PetscInt)std::lround(f["p"][3 * i]);
      for (PetscInt a = 0; a < 3; a++) {
        f["r"][3 * i + a] += displacement(id, a);

        // The step start must not be touched by the push
        if (step == 1) {
          PetscCheck(f["r_prev"][3 * i + a] == initial_position(id, a), PETSC_COMM_WORLD, PETSC_ERR_USER, "Previous position of particle %d is wrong", id);
          PetscCheck(f["p_prev"][3 * i + a] == f["p"][3 * i + a], PETSC_COMM_WORLD, PETSC_ERR_USER, "Previous momentum of particle %d is wrong", id);
        }
      }
    }
    PetscCall(storage.restore_attributes(f));

    PetscCall(storage.update_cells());
    PetscCall(check_particles(storage, step));
  }

  // Sorting permutes all the attributes together and gives the order of `World::s_g()`
  PetscCall(storage.sort());
  PetscCall(check_particles(storage, steps));

  PetscInt size;
  PetscCall(storage.get_size(&size));

  const std::vector<PetscInt>& offsets = storage.cell_offsets();
  PetscCheck((PetscInt)offsets.size() == world.size.elements_product() + 1 && offsets.back() == size, PETSC_COMM_WORLD, PETSC_ERR_USER, "Cell offsets do not cover the particles");

  PetscCall(storage.get_attributes(f));
  for (PetscInt g = 0; g < world.size.elements_product(); g++) {
    for (PetscInt i = offsets[g]; i < offsets[g + 1]; i++) {
      const PetscReal* r = &f["r"][3 * i];
      const PetscInt cell = world.s_g(          //
        FLOOR_STEP(r[X], dx) - world.start[X],  //
        FLOOR_STEP(r[Y], dy) - world.start[Y],  //
        FLOOR_STEP(r[Z], dz) - world.start[Z]);
      PetscCheck(cell == g, PETSC_COMM_WORLD, PETSC_ERR_USER, "Particle %d is in the block of cell %d, but belongs to %d", i, g, cell);
    }
  }
  PetscCall(storage.restore_attributes(f));

  PetscCall(storage.finalize());
  PetscCall(world.finalize());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
