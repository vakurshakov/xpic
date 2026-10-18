#include "src/diagnostics/field_view.h"

#include "src/utils/world.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing \'FieldView\' diagnostics: write, mpi-write, comparisons\n.";

static PetscErrorCode test_vec(DM da, Vec v)
{
  PetscFunctionBeginUser;
  std::filesystem::path out_dir = get_out_dir(__FILE__);

  Region region{
    .dim = 4,
    .dof = 3,
    .start = Vector4I{0, 0, 0, 0},
    .size = Vector4I{geom_nx, geom_ny, geom_nz, 3},
  };

  auto&& diag = FieldView::create(out_dir, da, v, region);
  PetscCall(diag->diagnose(0));
  PetscCall(diag->finalize());
  PetscCall(compare_with_vec(out_dir / "0000", da, v, 1e-6));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode test_vec_comp(DM da, Vec v)
{
  PetscFunctionBeginUser;
  std::filesystem::path out_dir = get_out_dir(__FILE__) / "component_z";

  Region region{
    .dim = 4,
    .dof = 3,
    .start = Vector4I{0, 0, 0, Z},
    .size = Vector4I{geom_nx, geom_ny, geom_nz, 1},
  };

  auto&& diag = FieldView::create(out_dir, da, v, region);
  PetscCall(diag->diagnose(0));
  PetscCall(diag->finalize());

  std::vector<float> written;
  PetscCall(read_floats(out_dir / "0000", geom_nx * geom_ny * geom_nz, written));

  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

  PetscReal**** arr;
  PetscCall(DMDAVecGetArrayDOFRead(da, v, &arr));
  for (PetscInt z = zs; z < zs + zm; ++z) {
    for (PetscInt y = ys; y < ys + ym; ++y) {
      for (PetscInt x = xs; x < xs + xm; ++x) {
        PetscCheck(equal_tol(written[(z * geom_ny + y) * geom_nx + x], arr[z][y][x][Z], 1e-6), PETSC_COMM_WORLD, PETSC_ERR_USER,
            "Written z-component differs from the field at (%d, %d, %d)", x, y, z);
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOFRead(da, v, &arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @todo for MPI, it should be tested for local (sub-regions) with different communicators
int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  World::set_geometry(REP3(5.0), 1.0, REP3(0.5), 1.0, 1.0);

  World world;
  PetscCall(world.initialize());

  Vec v;
  PetscCall(DMCreateGlobalVector(world.da, &v));
  PetscCall(VecSetRandom(v, nullptr));

  PetscCall(test_vec(world.da, v));
  PetscCall(test_vec_comp(world.da, v));

  PetscCall(VecDestroy(&v));
  PetscCall(world.finalize());

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
