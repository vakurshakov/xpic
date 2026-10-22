#include "src/commands/remove_particles.h"

#include "src/interfaces/simulation.h"
#include "src/diagnostics/energy.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing RemoveParticles with WithinBox and WithinCylinder: the geometry is\n"
  "shifted along each direction, the removed energy and count are checked.\n";

static constexpr PetscInt N = 8;

PetscErrorCode run();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));
  PetscCall(run());
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config()
{
  dx = dy = dz = 1.0;
  geom_nx = geom_ny = geom_nz = N;
  geom_x = geom_y = geom_z = N;

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
          {"Np", 1},
          {"n", +2.0},
          {"q", -1.0},
          {"m", +3.0},
          {"T", +0.1},
        },
      },
    },
  });
}

/// @brief `kept` is evaluated for each cell corner, as in `RemoveParticles`.
PetscErrorCode check_removal(interfaces::Particles& particles,
  RemoveParticles::Tester kept, const char* name)
{
  PetscFunctionBeginUser;
  const PetscReal m = particles.parameters.m;
  const PetscReal mpw = particles.parameters.mpw;

  for (auto& cell : particles.storage)
    cell.clear();

  PetscInt expected_removed = 0;
  PetscReal expected_energy = 0;

  for (PetscInt z = 0; z < N; z++) {
    for (PetscInt y = 0; y < N; y++) {
      for (PetscInt x = 0; x < N; x++) {
        // The point is in the cell center, the corner is tested
        Vector3R r(x + 0.5, y + 0.5, z + 0.5);

        // Each cell holds one particle with a distinct momentum, so that a
        // wrongly removed cell changes the energy. The components are dyadic,
        // so all energy sums are exact and independent of the summation order.
        Vector3R p((x + 1) / 8.0, (y + 1) / 16.0, (z + 1) / 32.0);

        PetscCall(particles.add_particle(Point{r, p}));

        if (!kept(Vector3R(x * dx, y * dy, z * dz))) {
          expected_removed++;
          expected_energy += Energy::get_kinetic(p, m, mpw);
        }
      }
    }
  }

  PetscCheck(expected_removed > 0 && expected_removed < POW3(N), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "%s: test is degenerate, %d cells are removed", name, expected_removed);

  RemoveParticles remove(particles, std::move(kept));
  PetscCall(remove.execute(0));

  PetscInt remained = 0;
  for (const auto& cell : particles.storage)
    remained += (PetscInt)cell.size();
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &remained, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));

  PetscCheck(remained == POW3(N) - expected_removed, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "%s: %d particles remained, expected %d", name, remained, POW3(N) - expected_removed);

  PetscCheck(remove.get_removed_energy() == expected_energy, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "%s: removed energy %g, expected %g", name, remove.get_removed_energy(), expected_energy);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode run()
{
  PetscFunctionBeginUser;
  overwrite_config();

  std::unique_ptr<interfaces::Simulation> simulation = build_simulation();
  PetscCall(simulation->initialize());

  auto& particles = simulation->get_named_particles("a");

  // The box leaves out a slab of two cells at the lower or upper side of
  // each axis, so the particles are removed only from that side.
  for (Axis axis : {X, Y, Z}) {
    for (PetscReal shift : {0, 2}) {
      BoxGeometry box{Vector3R(0, 0, 0), Vector3R(N, N, N)};
      box.min[axis] = shift;
      box.max[axis] = N - 2 + shift;

      auto name = std::format("box axis {} shift {}", (int)axis, shift);
      PetscCall(check_removal(particles, WithinBox(box), name.c_str()));
    }
  }

  // The cylinder is along z, it is moved along each direction. The cells with
  // |z - center| = height / 2 are not within, its radius is inclusive.
  const Vector3R center(4, 4, 4);
  for (Axis axis : {X, Y, Z}) {
    for (PetscReal shift : {-2, 2}) {
      CylinderGeometry cyl{center, 2.5, 4};
      cyl.center[axis] += shift;

      auto name = std::format("cylinder axis {} shift {}", (int)axis, shift);
      PetscCall(check_removal(particles, WithinCylinder(cyl), name.c_str()));
    }
  }

  PetscCall(simulation->finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
