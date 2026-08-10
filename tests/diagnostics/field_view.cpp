#include "src/diagnostics/field_view.h"

#include "src/utils/world.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing \'FieldView\' diagnostics: write, mpi-write, comparisons\n.";

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

  std::filesystem::path out_dir = get_out_dir(__FILE__);

  /// @note We should create the diagnostic within some scope to properly run the destructors.
  {
    Region region{
      .dim = 4,
      .dof = 3,
      .start = Vector4I{0, 0, 0, 0},
      .size = Vector4I{geom_nx, geom_ny, geom_nz, 3},
    };

    /// @todo for MPI, it should be tested for local (sub-regions) with different communicators
    auto&& diag = FieldView::create(out_dir, world.da, v, region);
    PetscCall(diag->diagnose(0));

    uintmax_t size = std::filesystem::file_size(out_dir / "0000");
    uintmax_t csize = sizeof(float) * region.size.elements_product();
    PetscCheck(size == csize, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Result file size should match the selected region");
    PetscCall(diag->finalize());
  }

  {
    Region region{
      .dim = 4,
      .dof = 3,
      .start = Vector4I{0, 0, 0, Z},
      .size = Vector4I{geom_nx, geom_ny, geom_nz, 1},
    };
    const std::filesystem::path component_out_dir = out_dir / "component_z";
    auto&& diag = FieldView::create(component_out_dir, world.da, v, region);
    PetscCall(diag->diagnose(0));

    const uintmax_t size =
      std::filesystem::file_size(component_out_dir / "0000");
    const uintmax_t expected_size =
      sizeof(float) * geom_nx * geom_ny * geom_nz;
    PetscCheck(size == expected_size, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "A one-component FieldView should write one float per cell");
    PetscCall(diag->finalize());
  }

  PetscCall(VecDestroy(&v));
  PetscCall(world.finalize());

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
