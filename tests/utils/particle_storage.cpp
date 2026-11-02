#include "src/utils/particle_storage.h"

#include "tests/common.h"

static constexpr char help[] =
  "Testing ParticleStorage: the previous attributes, the exchange of \n"
  "particles between ranks, the removal of particles (non-periodic   \n"
  "axes, `remove()`), uneven counts per rank in `grow()` and the     \n"
  "counting sort by cell.                                            \n";

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

static PetscInt cell_coordinate(PetscInt id, PetscInt axis)
{
  PetscInt cell = id / ppc;
  for (PetscInt a = 0; a < axis; a++)
    cell /= Geom_n[a];
  return cell % Geom_n[axis];
}

// Uneven case: the cells under the diagonal are filled, so the counts differ between ranks
static bool is_loaded(PetscInt id, bool uneven)
{
  return !uneven ||                     //
    2 *                                 //
      (cell_coordinate(id, X) +         //
        cell_coordinate(id, Y) +        //
        cell_coordinate(id, Z)) <       //
    Geom_n[X] + Geom_n[Y] + Geom_n[Z];  //
}

// A particle is dropped as soon as it leaves the global domain along a non-periodic axis
static bool is_alive(PetscInt id, PetscInt step, const World& world)
{
  for (PetscInt s = 0; s <= step; s++) {
    for (PetscInt a = 0; a < 3; a++) {
      const PetscReal x = initial_position(id, a) + s * displacement(id, a);
      if (world.bounds[a] != DM_BOUNDARY_PERIODIC && (0 > x || x >= Geom[a]))
        return false;
    }
  }
  return true;
}

static PetscErrorCode check_particles(ParticleStorage& storage, PetscInt step,
  const std::function<bool(PetscInt)>& expected_id)
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

  PetscInt expected = 0;
  PetscReal expected_sum = 0;
  for (PetscInt id = 0; id < ppc * Geom_n[X] * Geom_n[Y] * Geom_n[Z]; id++) {
    if (expected_id(id)) {
      expected++;
      expected_sum += id;
    }
  }
  PetscCheck(total == expected && total_sum == expected_sum, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Particles are lost or duplicated: %d of %d, sum of ids %g, expected %g", total, expected, total_sum, expected_sum);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode run(const std::array<DMBoundaryType, 3>& bounds, bool uneven)
{
  PetscFunctionBeginUser;
  World world;
  std::ranges::copy(bounds, world.bounds);
  PetscCall(world.initialize());

  ParticleStorage storage(world);
  PetscCall(storage.initialize());
  PetscCall(storage.register_attribute("r", 3));
  PetscCall(storage.register_attribute("p", 3));
  PetscCall(storage.register_previous("r"));
  PetscCall(storage.register_previous("p"));
  PetscCall(storage.commit_attributes());

  // Cell by cell, like `SetParticles` does: the counts differ between ranks
  ParticleStorage::Attributes f;
  PetscInt size = 0;
  for (PetscInt z = world.start[Z]; z < world.end[Z]; z++) {
    for (PetscInt y = world.start[Y]; y < world.end[Y]; y++) {
      for (PetscInt x = world.start[X]; x < world.end[X]; x++) {
        const PetscInt cell_id = ((z * Geom_n[Y] + y) * Geom_n[X] + x) * ppc;
        if (!is_loaded(cell_id, uneven))
          continue;

        PetscInt first;
        PetscCall(storage.grow(ppc, &first));
        PetscCheck(first == size, PETSC_COMM_WORLD, PETSC_ERR_USER, "`grow()` returned %d, expected %d", first, size);
        size += ppc;

        PetscCall(storage.get_attributes(f));
        for (PetscInt p = 0; p < ppc; p++) {
          const PetscInt i = first + p;
          const PetscInt id = cell_id + p;
          for (PetscInt a = 0; a < 3; a++)
            f["r"][3 * i + a] = initial_position(id, a);
          f["p"][3 * i + X] = id;
          f["p"][3 * i + Y] = 2.0 * id;
          f["p"][3 * i + Z] = -id;
        }
        PetscCall(storage.restore_attributes(f));
      }
    }
  }

  bool removed = false;
  auto expected = [&](PetscInt step)
  {
    return [&, step](PetscInt id)
    {
      return                          //
        is_loaded(id, uneven) &&      //
        is_alive(id, step, world) &&  //
        !(removed && id % 5 == 0);    //
    };
  };
  PetscCall(check_particles(storage, 0, expected(0)));

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
    PetscCall(check_particles(storage, step, expected(step)));
  }

  // Sorting permutes all the attributes together and gives the order of `World::s_g()`
  PetscCall(storage.sort());
  PetscCall(check_particles(storage, steps, expected(steps)));

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

  // Removal by index, some ranks may remove nothing
  std::vector<PetscInt> indices;
  PetscCall(storage.get_attributes(f));
  for (PetscInt i = 0; i < f.size; i++) {
    if ((PetscInt)std::lround(f["p"][3 * i]) % 5 == 0)
      indices.push_back(i);
  }
  PetscCall(storage.restore_attributes(f));
  PetscCall(storage.remove(indices));
  removed = true;
  PetscCall(check_particles(storage, steps, expected(steps)));

  // The storage stays usable after the removal
  PetscCall(storage.save_previous());
  PetscCall(storage.update_cells());
  PetscCall(check_particles(storage, steps, expected(steps)));

  PetscCall(storage.finalize());
  PetscCall(world.finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  // dx != dy != dz to catch axis mix-ups
  World::set_geometry(16.0, 9.0, 4.0, 1.0, 1.0, 0.5, 0.25, 1.0, 1.0);

  constexpr DMBoundaryType P = DM_BOUNDARY_PERIODIC;
  constexpr DMBoundaryType N = DM_BOUNDARY_NONE;
  PetscCall(run({P, P, P}, false));
  PetscCall(run({P, P, P}, true));
  PetscCall(run({N, N, N}, true));
  PetscCall(run({P, N, N}, false));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
