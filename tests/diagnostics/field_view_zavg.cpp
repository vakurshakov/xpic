#include "src/diagnostics/field_view_zavg.h"

#include "src/utils/world.h"
#include "tests/common.h"

static constexpr char help[] = "Testing \'FieldViewZAvg\' diagnostic\n.";

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  World::set_geometry(REP3(5.0), 1.0, REP3(0.5), 1.0, 1.0);

  World world;
  PetscCall(world.initialize());

  auto value = [](PetscInt x, PetscInt y, PetscInt z, PetscInt c)
  {
    return (PetscReal)(x + 12 * y + 123 * z + 1234 * c);
  };

  Vec v;
  PetscReal**** arr;
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(world.da, &xs, &ys, &zs, &xm, &ym, &zm));
  PetscCall(DMCreateGlobalVector(world.da, &v));
  PetscCall(DMDAVecGetArrayDOF(world.da, v, &arr));
  for (PetscInt z = zs; z < zs + zm; ++z) {
    for (PetscInt y = ys; y < ys + ym; ++y) {
      for (PetscInt x = xs; x < xs + xm; ++x) {
        for (PetscInt c = 0; c < 3; ++c) {
          arr[z][y][x][c] = value(x, y, z, c);
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(world.da, v, &arr));

  auto out_dir = get_out_dir(__FILE__);

  Region region{
    .dim = 4,
    .dof = 3,
    .start = Vector4I{0, 0, 0, 0},
    .size = Vector4I{geom_nx, geom_ny, 1, 3},
  };

  auto&& diag = FieldViewZAvg::create(out_dir, world.da, v, region);
  PetscCall(diag->diagnose(0));
  PetscCall(diag->finalize());

  const PetscInt n = geom_nx * geom_ny * 3;
  std::vector<float> written;
  PetscCall(read_floats(out_dir / "0000", n, written));

  for (PetscInt y = 0; y < geom_ny; ++y) {
    for (PetscInt x = 0; x < geom_nx; ++x) {
      for (PetscInt c = 0; c < 3; ++c) {
        PetscReal avg = 0;
        for (PetscInt z = 0; z < geom_nz; ++z)
          avg += value(x, y, z, c) / (PetscReal)geom_nz;

        PetscCheck(equal_tol(written[(y * geom_nx + x) * 3 + c], avg, 1e-7), PETSC_COMM_WORLD, PETSC_ERR_USER,
          "Written value differs from the z-average at (%d, %d, %d)", x, y, c);
      }
    }
  }

  PetscCall(VecDestroy(&v));
  PetscCall(world.finalize());

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
