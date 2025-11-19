#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic Esirkepov interpolation: simple field interpolation.\n"
  "Tests interpolation of arbitrary E, B, and gradB fields on a particle.\n";

using namespace drift_kinetic_test_utils;

constexpr Vector3R E0(1.0, 1.0, 1.0);
constexpr Vector3R B0(0.0, 0.0, 1.0);

void get_analytical_fields(
  const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = E0;
  B_p = B0 + r;
  gradB_p = {1.,1.,1.};
}

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config(5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

  FieldContext context;

  PetscCall(context.initialize([&](PetscInt i, PetscInt j, PetscInt k, Vector3R& E_g, Vector3R& B_g, Vector3R& gradB_g) {
    get_analytical_fields(Vector3R(i * dx, j * dy, k * dz), E_g, B_g, gradB_g);
  }));

  DriftKineticEsirkepov esirkepov(
    context.E_arr, context.B_arr, nullptr, context.gradB_arr);

  esirkepov.set_dBidrj(context.dBdx_arr, context.dBdy_arr, context.dBdz_arr);

  Vector3R pos_old(1, 1, 1);
  Vector3R pos_new(2, 2, 2);

  Vector3R E_p, B_p, gradB_p;
  esirkepov.interpolate(E_p, B_p, gradB_p, pos_new, pos_old);

  Vector3R E_e, B_e, gradB_e;
  get_analytical_fields(pos_new, E_e, B_e, gradB_e);

#if 0
  LOG("Test position: ({:.3f}, {:.3f}, {:.3f})", REP3_A(pos_new));
  LOG("Electric field:");
  LOG("  Expected:     ({:.6f}, {:.6f}, {:.6f})", REP3_A(E_e));
  LOG("  Interpolated: ({:.6f}, {:.6f}, {:.6f})", REP3_A(E_p));
  LOG("Magnetic field:");
  LOG("  Expected:     ({:.6f}, {:.6f}, {:.6f})", REP3_A(B_e));
  LOG("  Interpolated: ({:.6f}, {:.6f}, {:.6f})", REP3_A(B_p));
  LOG("Gradient B field:");
  LOG("  Expected:     ({:.6f}, {:.6f}, {:.6f})", REP3_A(gradB_e));
  LOG("  Interpolated: ({:.6f}, {:.6f}, {:.6f})", REP3_A(gradB_p));
#endif

  PetscCheck(equal_tol(E_p, E_e, PETSC_SMALL), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Electric field interpolation failed. Expected: (%.8e %.8e %.8e), got: (%.8e %.8e %.8e)", REP3_A(E_e), REP3_A(E_p));

  PetscCheck(equal_tol(B_p, B_e, PETSC_SMALL), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Magnetic field interpolation failed. Expected: (%.8e %.8e %.8e), got: (%.8e %.8e %.8e)", REP3_A(B_e), REP3_A(B_p));

  PetscCheck(equal_tol(gradB_p, gradB_e, PETSC_SMALL), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Gradient B field interpolation failed. Expected: (%.8e %.8e %.8e), got: (%.8e %.8e %.8e)", REP3_A(gradB_e), REP3_A(gradB_p));

  PetscCall(context.finalize());
  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
