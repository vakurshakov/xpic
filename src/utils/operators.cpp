#include "operators.h"

#include "src/pch.h"

Operator::Operator(DM da, PetscInt mdof, PetscInt ndof)
  : da_(da), mdof_(mdof), ndof_(ndof)
{
  PetscCallVoid(DMDAGetCorners(da_, REP3_A(&start_), REP3_A(&size_)));
}

/// @details For reference, see `DMCreateMatrix()`, `MatSetStencil()`, `MatSetValuesStencil()`.
/* static */ PetscErrorCode Operator::remap_stencil(
  DM da, PetscInt mdof, PetscInt size, MatStencil* idxm, PetscInt* jdxm)
{
  PetscFunctionBeginUser;
  PetscInt dims[4];
  PetscInt start[4];
  PetscCall(DMDAGetGhostCorners(da, REP3_AP(&start), REP3_AP(&dims)));

  dims[3] = mdof;
  start[3] = 0;

  auto noc = static_cast<PetscBool>(mdof == 1);
  PetscInt dim = 3 + static_cast<PetscInt>(!noc);

  auto in = (PetscInt*)idxm;

  for (PetscInt i = 0; i < size; ++i) {
    PetscInt tmp = *in++ - start[0];

    for (PetscInt j = 0; j < dim - 1; ++j)
      if ((*in++ - start[j + 1]) < 0 || tmp < 0)
        tmp = -1;
      else
        tmp = tmp * dims[j + 1] + *(in - 1) - start[j + 1];

    if (noc)
      in++;

    jdxm[i] = tmp;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscInt Operator::m_index(PetscInt x, PetscInt y, PetscInt z, PetscInt c) const
{
  return indexing::petsc_index(x, y, z, c, geom_nx, geom_ny, geom_nz, mdof_);
}

PetscInt Operator::n_index(PetscInt x, PetscInt y, PetscInt z, PetscInt c) const
{
  return indexing::petsc_index(x, y, z, c, geom_nx, geom_ny, geom_nz, ndof_);
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


FiniteDifferenceOperator::FiniteDifferenceOperator(
  DM da, PetscInt mdof, PetscInt ndof, const std::vector<PetscReal>& v)
  : Operator(da, mdof, ndof), values_(v)
{
}

PetscErrorCode FiniteDifferenceOperator::create_positive(Mat* mat)
{
  PetscFunctionBeginUser;
  PetscCall(create_matrix(mat));
  PetscCall(fill_matrix(*mat, Yee_shift::Positive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FiniteDifferenceOperator::create_negative(Mat* mat)
{
  PetscFunctionBeginUser;
  PetscCall(create_matrix(mat));
  PetscCall(fill_matrix(*mat, Yee_shift::Negative));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FiniteDifferenceOperator::create_matrix(Mat* mat)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreateMatrix(da_, mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FiniteDifferenceOperator::fill_matrix(Mat mat, Yee_shift shift)
{
  PetscFunctionBeginUser;
  const PetscInt chunk = values_.size();
  const PetscInt size = chunk * size_.elements_product();

  std::vector<MatStencil> coo_i(size);
  std::vector<MatStencil> coo_j(size);

  std::vector<PetscReal> coo_v;
  coo_v.reserve(size);

  for (PetscInt g = 0; g < size_.elements_product(); ++g) {
    PetscInt x = start_[X] + g % size_[X];
    PetscInt y = start_[Y] + (g / size_[X]) % size_[Y];
    PetscInt z = start_[Z] + (g / size_[X]) / size_[Y];

    MatStencil* coo_ci = coo_i.data() + g * chunk;
    MatStencil* coo_cj = coo_j.data() + g * chunk;

    // Periodic boundaries are handled by PETSc internally
    fill_stencil(shift, x, y, z, coo_ci, coo_cj);

    std::copy(values_.begin(), values_.end(), std::back_inserter(coo_v));
  }

  auto idxm = (PetscInt*)coo_i.data();
  auto idxn = (PetscInt*)coo_j.data();

  remap_stencil(da_, mdof_, size, coo_i.data(), idxm);
  remap_stencil(da_, ndof_, size, coo_j.data(), idxn);

  PetscCall(MatSetPreallocationCOOLocal(mat, size, idxm, idxn));
  PetscCall(MatSetValuesCOO(mat, coo_v.data(), ADD_VALUES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscInt FiniteDifferenceOperator::ind(PetscInt c, PetscInt i) const
{
  return c * (PetscInt)values_.size() / 3 + i;
}

std::tuple<REP3(PetscInt)> get_positive_offsets(PetscInt x, PetscInt y, PetscInt z)
{
  return std::make_tuple((x + 1), (y + 1), (z + 1));
}

std::tuple<REP3(PetscInt)> get_negative_offsets(PetscInt x, PetscInt y, PetscInt z)
{
  return std::make_tuple((x - 1), (y - 1), (z - 1));
}


Rotor::Rotor(DM da)
  : FiniteDifferenceOperator(da, 3, 3,
      {
        +1.0 / dz, -1.0 / dz, -1.0 / dy, +1.0 / dy,  //
        -1.0 / dz, +1.0 / dz, +1.0 / dx, -1.0 / dx,  //
        +1.0 / dy, -1.0 / dy, -1.0 / dx, +1.0 / dx,  //
      })
{
}

void Rotor::fill_stencil(Yee_shift shift, PetscInt xc, PetscInt yc, PetscInt zc,
  MatStencil* coo_i, MatStencil* coo_j) const
{
  for (PetscInt i = 0; i < (PetscInt)values_.size() / 3; ++i) {
    coo_i[ind(X, i)] = {zc, yc, xc, X};
    coo_i[ind(Y, i)] = {zc, yc, xc, Y};
    coo_i[ind(Z, i)] = {zc, yc, xc, Z};
  }

  switch (shift) {
    case Yee_shift::Positive: {
      auto&& [xp, yp, zp] = get_positive_offsets(xc, yc, zc);

      coo_j[ind(X, 0)] = {zp, yc, xc, Y};
      coo_j[ind(X, 1)] = {zc, yc, xc, Y};
      coo_j[ind(X, 2)] = {zc, yp, xc, Z};
      coo_j[ind(X, 3)] = {zc, yc, xc, Z};

      coo_j[ind(Y, 0)] = {zp, yc, xc, X};
      coo_j[ind(Y, 1)] = {zc, yc, xc, X};
      coo_j[ind(Y, 2)] = {zc, yc, xp, Z};
      coo_j[ind(Y, 3)] = {zc, yc, xc, Z};

      coo_j[ind(Z, 0)] = {zc, yp, xc, X};
      coo_j[ind(Z, 1)] = {zc, yc, xc, X};
      coo_j[ind(Z, 2)] = {zc, yc, xp, Y};
      coo_j[ind(Z, 3)] = {zc, yc, xc, Y};
      return;
    }

    case Yee_shift::Negative: {
      auto&& [xm, ym, zm] = get_negative_offsets(xc, yc, zc);

      coo_j[ind(X, 0)] = {zc, yc, xc, Y};
      coo_j[ind(X, 1)] = {zm, yc, xc, Y};
      coo_j[ind(X, 2)] = {zc, yc, xc, Z};
      coo_j[ind(X, 3)] = {zc, ym, xc, Z};

      coo_j[ind(Y, 0)] = {zc, yc, xc, X};
      coo_j[ind(Y, 1)] = {zm, yc, xc, X};
      coo_j[ind(Y, 2)] = {zc, yc, xc, Z};
      coo_j[ind(Y, 3)] = {zc, yc, xm, Z};

      coo_j[ind(Z, 0)] = {zc, yc, xc, X};
      coo_j[ind(Z, 1)] = {zc, ym, xc, X};
      coo_j[ind(Z, 2)] = {zc, yc, xc, Y};
      coo_j[ind(Z, 3)] = {zc, yc, xm, Y};
      return;
    }
  }
}


NonRectangularOperator::NonRectangularOperator(
  DM da, PetscInt mdof, PetscInt ndof, const std::vector<PetscReal>& v)
  : FiniteDifferenceOperator(da, mdof, ndof, v)
{
}

NonRectangularOperator::~NonRectangularOperator()
{
  PetscCallVoid(DMDestroy(&sda_));
}

PetscErrorCode NonRectangularOperator::create_matrix(Mat* mat)
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

PetscErrorCode NonRectangularOperator::create_scalar_da()
{
  PetscFunctionBeginUser;
  PetscInt dim;
  PetscInt g_size[3];
  PetscInt procs[3];
  PetscInt dof;
  PetscInt s;
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
  : NonRectangularOperator(da, 1, 3,
      {
        +1.0 / dx, -1.0 / dx,  //
        +1.0 / dy, -1.0 / dy,  //
        +1.0 / dz, -1.0 / dz,  //
      })
{
}

void Divergence::fill_stencil(Yee_shift shift, PetscInt xc, PetscInt yc,
  PetscInt zc, MatStencil* coo_i, MatStencil* coo_j) const
{
  for (PetscInt i = 0; i < (PetscInt)values_.size() / 3; ++i) {
    MatStencil s{zc, yc, xc};
    coo_i[ind(X, i)] = s;
    coo_i[ind(Y, i)] = s;
    coo_i[ind(Z, i)] = s;
  }

  switch (shift) {
    case Yee_shift::Positive: {
      auto&& [xp, yp, zp] = get_positive_offsets(xc, yc, zc);

      coo_j[ind(X, 0)] = {zc, yc, xp, X};
      coo_j[ind(X, 1)] = {zc, yc, xc, X};

      coo_j[ind(Y, 0)] = {zc, yp, xc, Y};
      coo_j[ind(Y, 1)] = {zc, yc, xc, Y};

      coo_j[ind(Z, 0)] = {zp, yc, xc, Z};
      coo_j[ind(Z, 1)] = {zc, yc, xc, Z};
      return;
    }

    case Yee_shift::Negative: {
      auto&& [xm, ym, zm] = get_negative_offsets(xc, yc, zc);

      coo_j[ind(X, 0)] = {zc, yc, xc, X};
      coo_j[ind(X, 1)] = {zc, yc, xm, X};

      coo_j[ind(Y, 0)] = {zc, yc, xc, Y};
      coo_j[ind(Y, 1)] = {zc, ym, xc, Y};

      coo_j[ind(Z, 0)] = {zc, yc, xc, Z};
      coo_j[ind(Z, 1)] = {zm, yc, xc, Z};
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
  : NonRectangularOperator(da, 3, 1,
      {
        +1.0 / dx, -1.0 / dx,  //
        +1.0 / dy, -1.0 / dy,  //
        +1.0 / dz, -1.0 / dz,  //
      })
{
}

void Gradient::fill_stencil(Yee_shift shift, PetscInt xc, PetscInt yc,
  PetscInt zc, MatStencil* coo_i, MatStencil* coo_j) const
{
  for (PetscInt i = 0; i < (PetscInt)values_.size() / 3; ++i) {
    coo_i[ind(X, i)] = {zc, yc, xc, X};
    coo_i[ind(Y, i)] = {zc, yc, xc, Y};
    coo_i[ind(Z, i)] = {zc, yc, xc, Z};
  }

  switch (shift) {
    case Yee_shift::Positive: {
      auto&& [xp, yp, zp] = get_positive_offsets(xc, yc, zc);

      coo_j[ind(X, 0)] = {zc, yc, xp};
      coo_j[ind(X, 1)] = {zc, yc, xc};

      coo_j[ind(Y, 0)] = {zc, yp, xc};
      coo_j[ind(Y, 1)] = {zc, yc, xc};

      coo_j[ind(Z, 0)] = {zp, yc, xc};
      coo_j[ind(Z, 1)] = {zc, yc, xc};
      return;
    }

    case Yee_shift::Negative: {
      auto&& [xm, ym, zm] = get_negative_offsets(xc, yc, zc);

      coo_j[ind(X, 0)] = {zc, yc, xc};
      coo_j[ind(X, 1)] = {zc, yc, xm};

      coo_j[ind(Y, 0)] = {zc, yc, xc};
      coo_j[ind(Y, 1)] = {zc, ym, xc};

      coo_j[ind(Z, 0)] = {zc, yc, xc};
      coo_j[ind(Z, 1)] = {zm, yc, xc};
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
