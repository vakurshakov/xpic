#include "operators.h"

#include "src/pch.h"

Operator::Operator(DM da) : da_(da) {
  /// @todo DMDAGetGhostCorners()?
  PetscCallVoid(DMDAGetCorners(da_, REP3_A(&start_), REP3_A(&size_)));
}

PetscInt Operator::index(PetscInt x, PetscInt y, PetscInt z, PetscInt c) const {
  return ((z * geom_ny + y) * geom_nx + x) * Vector3I::dim + c;
}


Identity::Identity(DM da) : Operator(da)
{
}

PetscErrorCode Identity::create(Mat* mat) const {
  PetscFunctionBeginUser;
  PetscInt ls = size_[X] * size_[Y] * size_[Z];
  PetscCall(MatCreateConstantDiagonal(PetscObjectComm((PetscObject)da_), ls, ls, PETSC_DETERMINE, PETSC_DETERMINE, 1.0, mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}


Finite_difference_operator::Finite_difference_operator(DM da, const std::vector<PetscReal>& v)
  : Operator(da), values(v)
{
}

PetscErrorCode Finite_difference_operator::create_positive(Mat* mat) const {
  PetscFunctionBeginUser;
  PetscCall(create_matrix(mat));
  PetscCall(fill_matrix(*mat, Shift::Positive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Finite_difference_operator::create_negative(Mat* mat) const {
  PetscFunctionBeginUser;
  PetscCall(create_matrix(mat));
  PetscCall(fill_matrix(*mat, Shift::Negative));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Finite_difference_operator::create_matrix(Mat* mat) const {
  PetscFunctionBeginUser;
  PetscCall(DMCreateMatrix(da_, mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Finite_difference_operator::fill_matrix(Mat mat, Shift s) const {
  PetscFunctionBeginUser;
  std::vector<MatStencil> row(Vector3R::dim);
  std::vector<MatStencil> col(Vector3R::dim * values.size());

  for (PetscInt z = start_[Z]; z < start_[Z] + size_[Z]; ++z) {
  for (PetscInt y = start_[Y]; y < start_[Y] + size_[Y]; ++y) {
  for (PetscInt x = start_[X]; x < start_[X] + size_[X]; ++x) {
    fill_stencil(s, x, y, z, row, col);

    // Periodic boundaries are handled by PETSc internally
    // We use `ADD_VALUES` to cancel out values in case of Nx = 1 (or Ny, Nz)
    PetscCall(MatSetValuesStencil(mat, row.size(), row.data(), col.size(), col.data(), values.data(), ADD_VALUES));
  }}}
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

std::tuple<REP3(PetscInt)> Finite_difference_operator::get_positive_offsets(PetscInt x, PetscInt y, PetscInt z) const {
  return std::make_tuple(
    (geom_nx > 1) ? (x + 1) : x,
    (geom_ny > 1) ? (y + 1) : y,
    (geom_nz > 1) ? (z + 1) : z
  );
}

std::tuple<REP3(PetscInt)> Finite_difference_operator::get_negative_offsets(PetscInt x, PetscInt y, PetscInt z) const {
  return std::make_tuple(
    (geom_nx > 1) ? (x - 1) : x,
    (geom_ny > 1) ? (y - 1) : y,
    (geom_nz > 1) ? (z - 1) : z
  );
}


Rotor::Rotor(DM da)
  : Finite_difference_operator(da,
    {+1.0 / dy, -1.0 / dy, -1.0 / dz, +1.0 / dz,
     +1.0 / dz, -1.0 / dz, -1.0 / dx, +1.0 / dx,
     +1.0 / dx, -1.0 / dx, -1.0 / dy, +1.0 / dy})
{
}

void Rotor::fill_stencil(Shift s, PetscInt x, PetscInt y, PetscInt z, std::vector<MatStencil>& row, std::vector<MatStencil>& col) const {
  switch (s) {
  case Shift::Positive: {
    auto&& [xp, yp, zp] = get_positive_offsets(x, y, z);

    row[0] = {z, y, x, X};
    col[0 + 0] = {z,  yp, x,  Z};
    col[0 + 1] = {z,  y,  x,  Z};
    col[0 + 2] = {zp, y,  x,  Y};
    col[0 + 3] = {z,  y,  x,  Y};

    row[1] = {z, y, x, Y};
    col[4 + 0] = {zp, y,  x,  X};
    col[4 + 1] = {z,  y,  x,  X};
    col[4 + 2] = {z,  y,  xp, Z};
    col[4 + 3] = {z,  y,  x,  Z};

    row[2] = {z, y, x, Z};
    col[8 + 0] = {z,  y,  xp, Y};
    col[8 + 1] = {z,  y,  x,  Y};
    col[8 + 2] = {z,  yp, x,  X};
    col[8 + 3] = {z,  y,  x,  X};
    return;
  }

  case Shift::Negative: {
    auto&& [xm, ym, zm] = get_negative_offsets(x, y, z);

    row[0] = {z, y, x, X};
    col[0 + 0] = {z,  y,  x,  Z};
    col[0 + 1] = {z,  ym, x,  Z};
    col[0 + 2] = {z,  y,  x,  Y};
    col[0 + 3] = {zm, y,  x,  Y};

    row[1] = {z, y, x, Y};
    col[4 + 0] = {z,  y,  x,  X};
    col[4 + 1] = {zm, y,  x,  X};
    col[4 + 2] = {z,  y,  x,  Z};
    col[4 + 3] = {z,  y,  xm, Z};

    row[2] = {z, y, x, Z};
    col[8 + 0] = {z,  y,  x,  Y};
    col[8 + 1] = {z,  y,  xm, Y};
    col[8 + 2] = {z,  y,  x,  X};
    col[8 + 3] = {z,  ym, x,  X};
    return;
  }}
}

Divergence::Divergence(DM da)
  : Finite_difference_operator(da,
    {+1.0 / dx, -1.0 / dx,
     +1.0 / dy, -1.0 / dy,
     +1.0 / dz, -1.0 / dz})
{
}

void Divergence::fill_stencil(Shift s, PetscInt x, PetscInt y, PetscInt z, std::vector<MatStencil>& row, std::vector<MatStencil>& col) const {
  switch (s) {
  case Shift::Positive: {
    auto&& [xp, yp, zp] = get_positive_offsets(x, y, z);

    row[0] = {z, y, x, 0};
    col[0 + 0] = {z,  y,  xp, X};
    col[0 + 1] = {z,  y,  x,  X};

    row[1] = row[0];
    col[2 + 0] = {z,  yp, x,  Y};
    col[2 + 1] = {z,  y,  x,  Y};

    row[2] = row[0];
    col[4 + 0] = {zp, y,  x,  Z};
    col[4 + 1] = {z,  y,  x,  Z};
    return;
  }

  case Shift::Negative: {
    auto&& [xm, ym, zm] = get_negative_offsets(x, y, z);

    row[0] = {z, y, x, 0};
    col[0 + 0] = {z,  y,  x,  X};
    col[0 + 1] = {z,  y,  xm, X};

    row[1] = row[0];
    col[2 + 0] = {z,  y,  x,  Y};
    col[2 + 1] = {z,  ym, x,  Y};

    row[2] = row[0];
    col[4 + 0] = {z,  y,  x,  Z};
    col[4 + 1] = {zm, y,  x,  Z};
    return;
  }}
}

PetscErrorCode Divergence::create_matrix(Mat* mat) const {
  PetscFunctionBeginUser;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)da_), mat));
  PetscCall(set_sizes(*mat));

  MatType mtype;
  PetscCall(DMGetMatType(da_, &mtype));
  PetscCall(MatSetType(*mat, mtype));

  PetscCall(MatSetUp(*mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Divergence::set_sizes(Mat mat) const {
  PetscFunctionBeginUser;
  PetscInt ls = size_[X] * size_[Y] * size_[Z];
  PetscCall(MatSetSizes(mat, ls, ls * Vector3R::dim, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscFunctionReturn(PETSC_SUCCESS);
}


Gradient::Gradient(DM da) : Divergence(da)
{
}

void Gradient::fill_stencil(Shift s, PetscInt x, PetscInt y, PetscInt z, std::vector<MatStencil>& row, std::vector<MatStencil>& col) const {
  switch (s) {
  case Shift::Positive: {
    auto&& [xp, yp, zp] = get_positive_offsets(x, y, z);

    row[0] = {z, y, x, X};
    col[0 + 0] = {z,  y,  xp, 0};
    col[0 + 1] = {z,  y,  x,  0};

    row[1] = {z, y, x, Y};
    col[2 + 0] = {z,  yp, x,  0};
    col[2 + 1] = {z,  y,  x,  0};

    row[2] = {z, y, x, Z};
    col[4 + 0] = {zp, y,  x,  0};
    col[4 + 1] = {z,  y,  x,  0};
    return;
  }

  case Shift::Negative: {
    auto&& [xm, ym, zm] = get_negative_offsets(x, y, z);

    row[0] = {z, y, x, X};
    col[0 + 0] = {z,  y,  x,  0};
    col[0 + 1] = {z,  y,  xm, 0};

    row[1] = {z, y, x, Y};
    col[2 + 0] = {z,  y,  x,  0};
    col[2 + 1] = {z,  ym, x,  0};

    row[2] = {z, y, x, Z};
    col[4 + 0] = {z,  y,  x,  0};
    col[4 + 1] = {zm, y,  x,  0};
    return;
  }}
}

PetscErrorCode Gradient::set_sizes(Mat mat) const {
  PetscFunctionBeginUser;
  PetscInt ls = size_[X] * size_[Y] * size_[Z];
  PetscCall(MatSetSizes(mat, ls * Vector3R::dim, ls, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

