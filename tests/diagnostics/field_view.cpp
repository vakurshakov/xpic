#include "src/diagnostics/field_view.h"

#include "src/utils/world.h"
#include "tests/diagnostics/common.h"

static char help[] =
  "Testing \'FieldView\' diagnostics: write, mpi-write, comparisons\n.";

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  World::set_geometry(REP3(5.0), 1.0, REP3(0.5), 1.0, 1.0);

  /// @todo different boundary conditions should be tested
  World world;
  PetscCall(world.initialize());

  Vec v;
  PetscCall(DMCreateGlobalVector(world.da, &v));
  PetscCall(VecSetRandom(v, nullptr));

  std::filesystem::path out_dir = get_out_dir(__FILE__);

  /// @note We should create the diagnostic within some scope to properly run the destructors.
  {
    /// @todo start, size should be tested, touch/no-touch.
    FieldView::Region region{
      .dim = 4,
      .dof = 3,
      .start = Vector4I{0, 0, 0, 0},
      .size = Vector4I{geom_nx, geom_ny, geom_nz, 3},
    };

    /// @todo for mpi, it should be tested for local (sub-regions) with different communicators
    auto&& diag = FieldView::create(out_dir, world.da, v, region);
    PetscCall(diag->diagnose(0));

    uintmax_t size = std::filesystem::file_size(out_dir.concat("/0"));
    uintmax_t csize = sizeof(float) * region.size.elements_product();
    PetscCheck(size == csize, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Result file size should match the selected region");
  }

  PetscCall(VecDestroy(&v));
  PetscCall(world.finalize());

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
