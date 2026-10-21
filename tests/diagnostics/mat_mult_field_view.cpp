#include "src/diagnostics/mat_mult_field_view.h"

#include "src/utils/operators.h"
#include "src/utils/world.h"
#include "tests/common.h"

static constexpr char help[] =
  "Testing \'MatMultFieldView\' diagnostics: vector and scalar results\n.";

static PetscErrorCode test(const std::filesystem::path& out_dir,  //
  DM result_da, Mat op, Vec source, const Region& region)
{
  PetscFunctionBeginUser;
  auto&& diag = MatMultFieldView::create(out_dir, result_da, op, source, region);
  PetscCall(diag->diagnose(0));
  PetscCall(diag->finalize());

  Vec expected;
  PetscCall(DMGetGlobalVector(result_da, &expected));
  PetscCall(MatMult(op, source, expected));

  PetscCall(compare_with_vec(out_dir / "0000", result_da, expected, 5e-7));
  PetscCall(DMRestoreGlobalVector(result_da, &expected));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  World::set_geometry(REP3(5.0), 1.0, REP3(0.5), 1.0, 1.0);

  World world;
  PetscCall(world.initialize());

  PetscCall(DMSetMatrixPreallocateOnly(world.da, PETSC_FALSE));
  PetscCall(DMSetMatrixPreallocateSkip(world.da, PETSC_TRUE));

  Mat rotB, divE;
  PetscCall(Rotor(world.da).create_negative(&rotB));
  PetscCall(Divergence(world.da).create_negative(&divE));

  Vec v;
  PetscCall(DMCreateGlobalVector(world.da, &v));
  PetscCall(VecSetRandom(v, nullptr));

  std::filesystem::path out_dir = get_out_dir(__FILE__);

  Region vector_region{
    .dim = 4,
    .dof = 3,
    .start = Vector4I{0, 0, 0, 0},
    .size = Vector4I{geom_nx, geom_ny, geom_nz, 3},
  };
  PetscCall(test(out_dir / "rotB", world.da, rotB, v, vector_region));

  Region scalar_region{
    .dim = 3,
    .dof = 1,
    .start = Vector4I{0, 0, 0, 0},
    .size = Vector4I{geom_nx, geom_ny, geom_nz, 1},
  };
  PetscCall(test(out_dir / "divE", world.da_rho, divE, v, scalar_region));

  PetscCall(MatDestroy(&rotB));
  PetscCall(MatDestroy(&divE));
  PetscCall(VecDestroy(&v));
  PetscCall(world.finalize());

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
