#include "operators.h"

#include "src/pch.h"

Operator::Operator(DM da, PetscInt mdof, PetscInt ndof)
  : da_(da), mdof_(mdof), ndof_(ndof)
{
  /// @note There is no need to use `DMDAGetGhostCorners()`, because here we
  /// only create operators that are used for solving equations (global), not to
  /// evaluate them. However, there should be a room in the stencil (ghost
  /// points) to use +-1 offsets.
  PetscCallVoid(DMDAGetCorners(da_, REP3_A(&start_), REP3_A(&size_)));
}

PetscInt Operator::index(
  PetscInt x, PetscInt y, PetscInt z, PetscInt c, PetscInt dof) const
{
  return ((z * geom_ny + y) * geom_nx + x) * dof + c;
}

PetscInt Operator::m_index(PetscInt x, PetscInt y, PetscInt z, PetscInt c) const
{
  return index(x, y, z, c, mdof_);
}

PetscInt Operator::n_index(PetscInt x, PetscInt y, PetscInt z, PetscInt c) const
{
  return index(x, y, z, c, ndof_);
}


Identity::Identity(DM da)
  : Operator(da)
{
}

PetscErrorCode Identity::create(Mat* mat) const
{
  PetscFunctionBeginUser;
  PetscInt ls = size_[X] * size_[Y] * size_[Z];  // without ghost cells
  PetscCall(MatCreateConstantDiagonal(PetscObjectComm((PetscObject)da_), ls, ls, PETSC_DETERMINE, PETSC_DETERMINE, 1.0, mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}


Finite_difference_operator::Finite_difference_operator(
  DM da, PetscInt mdof, PetscInt ndof, const std::vector<PetscReal>& v)
  : Operator(da, mdof, ndof), values_(v)
{
}

PetscErrorCode Finite_difference_operator::create_positive(Mat* mat)
{
  PetscFunctionBeginUser;
  PetscCall(create_matrix(mat));
  PetscCall(fill_matrix(*mat, Yee_shift::Positive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Finite_difference_operator::create_negative(Mat* mat)
{
  PetscFunctionBeginUser;
  PetscCall(create_matrix(mat));
  PetscCall(fill_matrix(*mat, Yee_shift::Negative));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Finite_difference_operator::create_matrix(Mat* mat)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreateMatrix(da_, mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Finite_difference_operator::fill_matrix(Mat mat, Yee_shift shift)
{
  PetscFunctionBeginUser;

  std::vector<MatStencil> row(3);
  std::vector<MatStencil> col(values_.size());

  const PetscInt chunk_size = values_.size() / 3;

  // clang-format off
  for (PetscInt z = start_[Z]; z < start_[Z] + size_[Z]; ++z) {
  for (PetscInt y = start_[Y]; y < start_[Y] + size_[Y]; ++y) {
  for (PetscInt x = start_[X]; x < start_[X] + size_[X]; ++x) {
    fill_stencil(shift, x, y, z, row, col);

    // Periodic boundaries are handled by PETSc internally
    for (PetscInt c = 0; c < 3; ++c) {
      PetscCall(mat_set_values_stencil(mat, 1, &row[c], chunk_size, (col.data() + chunk_size * c), (values_.data() + chunk_size * c), ADD_VALUES));
    }
  }}}
  // clang-format on
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @details For reference, see `DMCreateMatrix()`, `DMCreateMatrix_DA()`
/// `MatSetStencil()`, `MatSetValuesStencil()`
PetscErrorCode Finite_difference_operator::mat_set_values_stencil(Mat mat,
  PetscInt m, const MatStencil idxm[], PetscInt n, const MatStencil idxn[],
  const PetscScalar v[], InsertMode addv) const
{
  PetscFunctionBegin;
  static const PetscInt MAX_CHUNK_SIZE = 4;
  PetscCheck(m < MAX_CHUNK_SIZE, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Inserted columns number %" PetscInt_FMT ", is greater than MAX_CHUNK_SIZE %" PetscInt_FMT, m, MAX_CHUNK_SIZE);

  // Inserting one row at a time, cols are limited by `MAX_CHUNK_SIZE`
  PetscInt jdxm[1];
  PetscInt jdxn[MAX_CHUNK_SIZE];

  PetscInt dims[4];
  PetscInt starts[4];
  PetscCall(DMDAGetGhostCorners(da_, REP3_AP(&starts), REP3_AP(&dims)));

  auto remap_indices = [&starts, &dims](PetscInt mdof, PetscInt m,
                         const MatStencil idxm[], PetscInt* jdxm) {
    dims[3] = mdof;
    starts[3] = 0;

    PetscBool noc = (PetscBool)(mdof == 1);
    PetscInt dim = 3 + (PetscInt)!noc;

    PetscInt i, j, tmp;
    PetscInt* dxm = (PetscInt*)idxm;

    for (i = 0; i < m; ++i) {
      tmp = *dxm++ - starts[0];
      for (j = 0; j < dim - 1; ++j)
        if ((*dxm++ - starts[j + 1]) < 0 || tmp < 0)
          tmp = -1;
        else
          tmp = tmp * dims[j + 1] + *(dxm - 1) - starts[j + 1];
      if (noc)
        dxm++;
      jdxm[i] = tmp;
    }
  };

  remap_indices(mdof_, m, idxm, jdxm);
  remap_indices(ndof_, n, idxn, jdxn);

  PetscCall(MatSetValuesLocal(mat, m, jdxm, n, jdxn, v, addv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

std::tuple<REP3(PetscInt)> get_positive_offsets(PetscInt x, PetscInt y, PetscInt z)
{
  return std::make_tuple(         //
    (geom_nx > 1) ? (x + 1) : x,  //
    (geom_ny > 1) ? (y + 1) : y,  //
    (geom_nz > 1) ? (z + 1) : z);
}

std::tuple<REP3(PetscInt)> get_negative_offsets(PetscInt x, PetscInt y, PetscInt z)
{
  return std::make_tuple(         //
    (geom_nx > 1) ? (x - 1) : x,  //
    (geom_ny > 1) ? (y - 1) : y,  //
    (geom_nz > 1) ? (z - 1) : z);
}


Rotor::Rotor(DM da)
  : Finite_difference_operator(da, 3, 3,
      {+1.0 / dy, -1.0 / dy, -1.0 / dz, +1.0 / dz,   //
        +1.0 / dz, -1.0 / dz, -1.0 / dx, +1.0 / dx,  //
        +1.0 / dx, -1.0 / dx, -1.0 / dy, +1.0 / dy})
{
}

void Rotor::fill_stencil(Yee_shift shift, PetscInt xc, PetscInt yc, PetscInt zc,
  std::vector<MatStencil>& row, std::vector<MatStencil>& col) const
{
  switch (shift) {
    case Yee_shift::Positive: {
      auto&& [xp, yp, zp] = get_positive_offsets(xc, yc, zc);

      row[0] = {zc, yc, xc, X};
      col[0 + 0] = {zc, yp, xc, Z};
      col[0 + 1] = {zc, yc, xc, Z};
      col[0 + 2] = {zp, yc, xc, Y};
      col[0 + 3] = {zc, yc, xc, Y};

      row[1] = {zc, yc, xc, Y};
      col[4 + 0] = {zp, yc, xc, X};
      col[4 + 1] = {zc, yc, xc, X};
      col[4 + 2] = {zc, yc, xp, Z};
      col[4 + 3] = {zc, yc, xc, Z};

      row[2] = {zc, yc, xc, Z};
      col[8 + 0] = {zc, yc, xp, Y};
      col[8 + 1] = {zc, yc, xc, Y};
      col[8 + 2] = {zc, yp, xc, X};
      col[8 + 3] = {zc, yc, xc, X};
      return;
    }

    case Yee_shift::Negative: {
      auto&& [xm, ym, zm] = get_negative_offsets(xc, yc, zc);

      row[0] = {zc, yc, xc, X};
      col[0 + 0] = {zc, yc, xc, Z};
      col[0 + 1] = {zc, ym, xc, Z};
      col[0 + 2] = {zc, yc, xc, Y};
      col[0 + 3] = {zm, yc, xc, Y};

      row[1] = {zc, yc, xc, Y};
      col[4 + 0] = {zc, yc, xc, X};
      col[4 + 1] = {zm, yc, xc, X};
      col[4 + 2] = {zc, yc, xc, Z};
      col[4 + 3] = {zc, yc, xm, Z};

      row[2] = {zc, yc, xc, Z};
      col[8 + 0] = {zc, yc, xc, Y};
      col[8 + 1] = {zc, yc, xm, Y};
      col[8 + 2] = {zc, yc, xc, X};
      col[8 + 3] = {zc, ym, xc, X};
      return;
    }
  }
}


Non_rectangular_operator::Non_rectangular_operator(
  DM da, PetscInt mdof, PetscInt ndof, const std::vector<PetscReal>& v)
  : Finite_difference_operator(da, mdof, ndof, v)
{
}

Non_rectangular_operator::~Non_rectangular_operator()
{
  PetscCallVoid(DMDestroy(&sda_));
}

PetscErrorCode Non_rectangular_operator::create_matrix(Mat* mat)
{
  PetscFunctionBeginUser;
  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)da_, &comm));
  PetscCall(MatCreate(comm, mat));

  MatType mtype;
  PetscCall(DMGetMatType(da_, &mtype));
  PetscCall(MatSetType(*mat, mtype));

  PetscCall(create_scalar_da());
  PetscCall(set_sizes_and_ltog(*mat));

  PetscCall(MatSetUp(*mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Non_rectangular_operator::create_scalar_da()
{
  PetscFunctionBeginUser;
  PetscInt dim, g_size[3], procs[3], dof, s;
  DMBoundaryType bounds[3];
  DMDAStencilType st;
  PetscCall(DMDAGetInfo(da_, &dim, REP3_A(&g_size), REP3_A(&procs), &dof, &s, REP3_A(&bounds), &st));

  const PetscInt* ownerships[3];
  PetscCall(DMDAGetOwnershipRanges(da_, REP3_A(&ownerships)));

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)da_, &comm));

  PetscCall(DMDACreate3d(comm, REP3_A(bounds), st, REP3_A(g_size), REP3_A(procs), 1, s, REP3_A(ownerships), &sda_));
  PetscCall(DMSetUp(sda_));

  l_size_ = size_[X] * size_[Y] * size_[Z];  // without ghost cells
  PetscCall(DMGetLocalToGlobalMapping(da_, &v_ltog_));
  PetscCall(DMGetLocalToGlobalMapping(sda_, &s_ltog_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


Divergence::Divergence(DM da)
  : Non_rectangular_operator(da, 1, 3,
      {+1.0 / dx, -1.0 / dx,   //
        +1.0 / dy, -1.0 / dy,  //
        +1.0 / dz, -1.0 / dz})
{
}

void Divergence::fill_stencil(Yee_shift shift, PetscInt xc, PetscInt yc,
  PetscInt zc, std::vector<MatStencil>& row, std::vector<MatStencil>& col) const
{
  switch (shift) {
    case Yee_shift::Positive: {
      auto&& [xp, yp, zp] = get_positive_offsets(xc, yc, zc);

      row[0] = {zc, yc, xc, 0};
      col[0 + 0] = {zc, yc, xp, X};
      col[0 + 1] = {zc, yc, xc, X};

      row[1] = row[0];
      col[2 + 0] = {zc, yp, xc, Y};
      col[2 + 1] = {zc, yc, xc, Y};

      row[2] = row[0];
      col[4 + 0] = {zp, yc, xc, Z};
      col[4 + 1] = {zc, yc, xc, Z};
      return;
    }

    case Yee_shift::Negative: {
      auto&& [xm, ym, zm] = get_negative_offsets(xc, yc, zc);

      row[0] = {zc, yc, xc, 0};
      col[0 + 0] = {zc, yc, xc, X};
      col[0 + 1] = {zc, yc, xm, X};

      row[1] = row[0];
      col[2 + 0] = {zc, yc, xc, Y};
      col[2 + 1] = {zc, ym, xc, Y};

      row[2] = row[0];
      col[4 + 0] = {zc, yc, xc, Z};
      col[4 + 1] = {zm, yc, xc, Z};
      return;
    }
  }
}

PetscErrorCode Divergence::set_sizes_and_ltog(Mat mat) const
{
  PetscFunctionBeginUser;
  PetscCall(MatSetSizes(mat, l_size_, l_size_ * 3, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetLocalToGlobalMapping(mat, s_ltog_, v_ltog_));
  PetscFunctionReturn(PETSC_SUCCESS);
}


Gradient::Gradient(DM da)
  : Non_rectangular_operator(da, 3, 1,
      {+1.0 / dx, -1.0 / dx,   //
        +1.0 / dy, -1.0 / dy,  //
        +1.0 / dz, -1.0 / dz})
{
}

void Gradient::fill_stencil(Yee_shift shift, PetscInt xc, PetscInt yc,
  PetscInt zc, std::vector<MatStencil>& row, std::vector<MatStencil>& col) const
{
  switch (shift) {
    case Yee_shift::Positive: {
      auto&& [xp, yp, zp] = get_positive_offsets(xc, yc, zc);

      row[0] = {zc, yc, xc, X};
      col[0 + 0] = {zc, yc, xp, 0};
      col[0 + 1] = {zc, yc, xc, 0};

      row[1] = {zc, yc, xc, Y};
      col[2 + 0] = {zc, yp, xc, 0};
      col[2 + 1] = {zc, yc, xc, 0};

      row[2] = {zc, yc, xc, Z};
      col[4 + 0] = {zp, yc, xc, 0};
      col[4 + 1] = {zc, yc, xc, 0};
      return;
    }

    case Yee_shift::Negative: {
      auto&& [xm, ym, zm] = get_negative_offsets(xc, yc, zc);

      row[0] = {zc, yc, xc, X};
      col[0 + 0] = {zc, yc, xc, 0};
      col[0 + 1] = {zc, yc, xm, 0};

      row[1] = {zc, yc, xc, Y};
      col[2 + 0] = {zc, yc, xc, 0};
      col[2 + 1] = {zc, ym, xc, 0};

      row[2] = {zc, yc, xc, Z};
      col[4 + 0] = {zc, yc, xc, 0};
      col[4 + 1] = {zm, yc, xc, 0};
      return;
    }
  }
}

PetscErrorCode Gradient::set_sizes_and_ltog(Mat mat) const
{
  PetscFunctionBeginUser;
  PetscCall(MatSetSizes(mat, l_size_ * 3, l_size_, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetLocalToGlobalMapping(mat, v_ltog_, s_ltog_));
  PetscFunctionReturn(PETSC_SUCCESS);
}
