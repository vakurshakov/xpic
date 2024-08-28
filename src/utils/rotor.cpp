#include "rotor.h"

#include "src/pch.h"
#include "src/utils/vector3.h"

Rotor::Rotor(DM da) :
  values_x{+1.0 / dy, -1.0 / dy, -1.0 / dz, +1.0 / dz},
  values_y{+1.0 / dz, -1.0 / dz, -1.0 / dx, +1.0 / dx},
  values_z{+1.0 / dx, -1.0 / dx, -1.0 / dy, +1.0 / dy},
  da_(da)
{
  /// @todo DMDAGetGhostCorners()?
  PetscCallVoid(DMDAGetCorners(da_, REP3_A(&start_), REP3_A(&size_)));
}

PetscErrorCode Rotor::set_positive(Mat mat) {
  PetscFunctionBeginUser;
  MatStencil row, col[4];
  for (PetscInt z = start_[Z]; z < start_[Z] + size_[Z]; ++z) {
  for (PetscInt y = start_[Y]; y < start_[Y] + size_[Y]; ++y) {
  for (PetscInt x = start_[X]; x < start_[X] + size_[X]; ++x) {
    // Periodic boundaries are handled by PETSc internally
    PetscInt xp = (geom_nx > 1) ? (x + 1) : x;
    PetscInt yp = (geom_ny > 1) ? (y + 1) : y;
    PetscInt zp = (geom_nz > 1) ? (z + 1) : z;

    row = {z, y, x, X};
    col[0] = {z,  yp, x,  Z};
    col[1] = {z,  y,  x,  Z};
    col[2] = {zp, y,  x,  Y};
    col[3] = {z,  y,  x,  Y};
    // We use `ADD_VALUES` to cancel out values in case of Nx = 1 (or Ny, Nz)
    PetscCall(MatSetValuesStencil(mat, 1, &row, 4, col, values_x, ADD_VALUES));

    row = {z, y, x, Y};
    col[0] = {zp, y,  x,  X};
    col[1] = {z,  y,  x,  X};
    col[2] = {z,  y,  xp, Z};
    col[3] = {z,  y,  x,  Z};
    PetscCall(MatSetValuesStencil(mat, 1, &row, 4, col, values_y, ADD_VALUES));

    row = {z, y, x, Z};
    col[0] = {z,  y,  xp, Y};
    col[1] = {z,  y,  x,  Y};
    col[2] = {z,  yp, x,  X};
    col[3] = {z,  y,  x,  X};
    PetscCall(MatSetValuesStencil(mat, 1, &row, 4, col, values_z, ADD_VALUES));
  }}}
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Rotor::set_negative(Mat mat) {
  PetscFunctionBeginUser;
  MatStencil row, col[4];
  for (PetscInt z = start_[Z]; z < start_[Z] + size_[Z]; ++z) {
  for (PetscInt y = start_[Y]; y < start_[Y] + size_[Y]; ++y) {
  for (PetscInt x = start_[X]; x < start_[X] + size_[X]; ++x) {
    PetscInt xm = (geom_nx > 1) ? (x - 1) : x;
    PetscInt ym = (geom_ny > 1) ? (y - 1) : y;
    PetscInt zm = (geom_nz > 1) ? (z - 1) : z;

    row = {z, y, x, X};
    col[0] = {z,  y,  x,  Z};
    col[1] = {z,  ym, x,  Z};
    col[2] = {z,  y,  x,  Y};
    col[3] = {zm, y,  x,  Y};
    PetscCall(MatSetValuesStencil(mat, 1, &row, 4, col, values_x, ADD_VALUES));

    row = {z, y, x, Y};
    col[0] = {z,  y,  x,  X};
    col[1] = {zm, y,  x,  X};
    col[2] = {z,  y,  x,  Z};
    col[3] = {z,  y,  xm, Z};
    PetscCall(MatSetValuesStencil(mat, 1, &row, 4, col, values_y, ADD_VALUES));

    row = {z, y, x, Z};
    col[0] = {z,  y,  x,  Y};
    col[0] = {z,  y,  xm, Y};
    col[0] = {z,  y,  x,  X};
    col[0] = {z,  ym, x,  X};
    PetscCall(MatSetValuesStencil(mat, 1, &row, 4, col, values_z, ADD_VALUES));
  }}}
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscInt Rotor::index(PetscInt x, PetscInt y, PetscInt z, PetscInt c) {
  return ((z * geom_ny + y) * geom_nx + x) * Vector3I::dim + c;
}
