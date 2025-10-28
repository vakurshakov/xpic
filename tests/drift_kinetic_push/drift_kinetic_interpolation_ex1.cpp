#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic Esirkepov interpolation: simple field interpolation.\n"
  "Tests interpolation of arbitrary E, B, and gradB fields on a particle.\n";

void get_analytical_fields(
  Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = Vector3R(1.0, 1.0, 1.0);

  B_p = Vector3R(0.0, 0.0, 1.0);

  gradB_p = r;
}

PetscErrorCode initialize_grid_fields(DM da, Vec E_vec, Vec B_vec, Vec gradB_vec)
{
  PetscFunctionBeginUser;

  Vector3R ***E_arr, ***B_arr, ***gradB_arr;

  PetscCall(DMDAVecGetArrayWrite(da, E_vec, &E_arr));
  PetscCall(DMDAVecGetArrayWrite(da, B_vec, &B_arr));
  PetscCall(DMDAVecGetArrayWrite(da, gradB_vec, &gradB_arr));

  Vector3I start, size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  for (PetscInt k = start[Z]; k < start[Z] + size[Z]; k++) {
    for (PetscInt j = start[Y]; j < start[Y] + size[Y]; j++) {
      for (PetscInt i = start[X]; i < start[X] + size[X]; i++) {
        Vector3R r(i, j, k);

        Vector3R E_analytical, B_analytical, gradB_analytical;
        get_analytical_fields(r, E_analytical, B_analytical, gradB_analytical);

        E_arr[k][j][i] = E_analytical;
        B_arr[k][j][i] = B_analytical;
        gradB_arr[k][j][i] = gradB_analytical;
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, E_vec, &E_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, B_vec, &B_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, gradB_vec, &gradB_arr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  World::set_geometry(10.0, 10.0, 10.0, 1.0, 1., 1., 1., 1.0, 1.0);

  World world;
  PetscCall(world.initialize());

  Vec E_vec, B_vec, gradB_vec;
  PetscCall(DMCreateGlobalVector(world.da, &E_vec));
  PetscCall(DMCreateGlobalVector(world.da, &B_vec));
  PetscCall(DMCreateGlobalVector(world.da, &gradB_vec));

  PetscCall(initialize_grid_fields(world.da, E_vec, B_vec, gradB_vec));

  Vector3R ***E_arr, ***B_arr, ***gradB_arr;
  PetscCall(DMDAVecGetArrayRead(world.da, E_vec, &E_arr));
  PetscCall(DMDAVecGetArrayRead(world.da, B_vec, &B_arr));
  PetscCall(DMDAVecGetArrayRead(world.da, gradB_vec, &gradB_arr));

  std::unique_ptr<DriftKineticEsirkepov> esirkepov =
    std::make_unique<DriftKineticEsirkepov>(E_arr, B_arr, nullptr, gradB_arr);

  Vector3R test_position_old(1, 1, 1);
  Vector3R test_position_new(2, 2, 2);

  Vector3R E_interpolated, B_interpolated, gradB_interpolated;

  esirkepov->interpolate(E_interpolated, B_interpolated, gradB_interpolated,
    test_position_new, test_position_old);

  Vector3R E_p = E_interpolated;
  Vector3R B_p = B_interpolated;
  Vector3R gradB_p = gradB_interpolated;

  Vector3R E_expected, B_expected, gradB_expected;
  get_analytical_fields(
    test_position_new, E_expected, B_expected, gradB_expected);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Test position: (%.3f, %.3f, %.3f)\n",
                       REP3_A(test_position_new)));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nElectric field:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Expected:     (%.6f, %.6f, %.6f)\n",
                       REP3_A(E_expected)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Interpolated: (%.6f, %.6f, %.6f)\n",
                       REP3_A(E_p)));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nMagnetic field:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Expected:     (%.6f, %.6f, %.6f)\n",
                       REP3_A(B_expected)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Interpolated: (%.6f, %.6f, %.6f)\n",
                       REP3_A(B_p)));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nGradient B field:\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Expected:     (%.6f, %.6f, %.6f)\n",
                       REP3_A(gradB_expected)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  Interpolated: (%.6f, %.6f, %.6f)\n",
                       REP3_A(gradB_p)));

  constexpr PetscReal tolerance = 1e-10;

  PetscCheck(equal_tol(E_interpolated, E_expected, tolerance), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Electric field interpolation failed. Expected: (%.8e %.8e %.8e), got: (%.8e %.8e %.8e)",
    REP3_A(E_expected), REP3_A(E_interpolated));

  PetscCheck(equal_tol(B_interpolated, B_expected, tolerance), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Magnetic field interpolation failed. Expected: (%.8e %.8e %.8e), got: (%.8e %.8e %.8e)",
    REP3_A(B_expected), REP3_A(B_interpolated));

  PetscCheck(equal_tol(gradB_interpolated, gradB_expected, tolerance), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Gradient B field interpolation failed. Expected: (%.8e %.8e %.8e), got: (%.8e %.8e %.8e)",
    REP3_A(gradB_expected), REP3_A(gradB_interpolated));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nAll interpolation tests passed!\n"));

  PetscCall(DMDAVecRestoreArrayRead(world.da, E_vec, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(world.da, B_vec, &B_arr));
  PetscCall(DMDAVecRestoreArrayRead(world.da, gradB_vec, &gradB_arr));

  PetscCall(VecDestroy(&E_vec));
  PetscCall(VecDestroy(&B_vec));
  PetscCall(VecDestroy(&gradB_vec));

  PetscCall(world.finalize());
  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
