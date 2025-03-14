#include "src/diagnostics/field_view.h"

#include "src/utils/world.h"

static char help[] = "\n\n.";

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  World::set_geometry(10, 10, 10, 1, 0.5, 0.5, 0.5, 1.0, 1.0);

  /// @todo different boundary conditions should be tested, touch/no-touch.
  World world;
  PetscCall(world.initialize());

  Vec v;
  PetscCall(DMCreateGlobalVector(world.da, &v));
  PetscCall(VecSetRandom(v, nullptr));

  /// @todo start, size should be tested
  FieldView::Region region{
    .dim = 4,
    .dof = 3,
    .start = Vector4I{0, 0, 0, 0},
    .size = Vector4I{geom_nx, geom_ny, geom_nz, 3},
  };

  std::filesystem::path out_dir(__FILE__);
  out_dir.replace_extension("");
  out_dir = std::format("{}/output/{}/", //
    out_dir.parent_path().c_str(), out_dir.filename().c_str());

  /// @note We should create the diagnostic within some scope to properly run the destructors.
  {
    /// @todo for mpi, it should be tested for local (sub-regions) with different communicators
    auto&& diag = FieldView::create(out_dir, world.da, v, region);
    PetscCall(diag->diagnose(0));
  }

  auto size = std::filesystem::file_size(out_dir.concat("/0.bin"));
  PetscCheck(size == sizeof(float) * region.size.elements_product(), PETSC_COMM_WORLD, PETSC_ERR_USER, "Result file size should match the selected region");

  PetscCall(VecDestroy(&v));
  PetscCall(world.finalize());

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
