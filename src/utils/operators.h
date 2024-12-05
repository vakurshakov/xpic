#ifndef SRC_UTILS_OPERATORS_H
#define SRC_UTILS_OPERATORS_H

#include <petscdmda.h>
#include <petscis.h>
#include <petscmat.h>

#include "src/utils/utils.h"

/// @brief Utility class to create constant operators on a `DMDA` grid.
class Operator {
public:
  virtual ~Operator() = default;

protected:
  Operator(DM da, PetscInt mdof = 3, PetscInt ndof = 3);

  PetscInt m_index(PetscInt z, PetscInt y, PetscInt x, PetscInt c) const;
  PetscInt n_index(PetscInt z, PetscInt y, PetscInt x, PetscInt c) const;

  DM da_;
  PetscInt start_[3], size_[3];
  PetscInt mdof_, ndof_;
};


class Identity final : public Operator {
public:
  Identity(DM da);
  PetscErrorCode create(Mat* mat) const;
};


/**
 * @brief This structure serves as an abstraction of derivatives in finite-
 * difference approximation. Yee stencil is used, so each operator can be
 * represented with both positive/negative offsets.
 */
class FiniteDifferenceOperator : public Operator {
public:
  PetscErrorCode create_positive(Mat* mat);
  PetscErrorCode create_negative(Mat* mat);

protected:
  /// @brief Can not be created explicitly as it is abstract operator.
  FiniteDifferenceOperator(
    DM da, PetscInt mdof, PetscInt ndof, const std::vector<PetscReal>& v);

  enum class Yee_shift {
    Positive,
    Negative,
  };

  virtual PetscErrorCode create_matrix(Mat* mat);
  PetscErrorCode fill_matrix(Mat mat, Yee_shift sh);

  /// @brief Specifies the stencil for each point `(x, y, z)` in space, after
  /// that whole chunk of `values` will be inserted into the matrix at once.
  virtual void fill_stencil(Yee_shift sh, PetscInt x, PetscInt y, PetscInt z,
    std::vector<MatStencil>& row, std::vector<MatStencil>& col) const = 0;

  /// @brief Almost identical to `MatSetValuesStencil()`,
  /// but can use different dof for rows and columns.
  PetscErrorCode mat_set_values_stencil(Mat, PetscInt, const MatStencil[],
    PetscInt, const MatStencil[], const PetscScalar[], InsertMode) const;

  const std::vector<PetscReal> values_;
};


class Rotor final : public FiniteDifferenceOperator {
public:
  Rotor(DM da);

private:
  void fill_stencil(Yee_shift s, PetscInt x, PetscInt y, PetscInt z,
    std::vector<MatStencil>& row, std::vector<MatStencil>& col) const override;
};


class NonRectangularOperator : public FiniteDifferenceOperator {
public:
  ~NonRectangularOperator() override;

protected:
  NonRectangularOperator(
    DM da, PetscInt mdof, PetscInt ndof, const std::vector<PetscReal>& v);

  PetscErrorCode create_matrix(Mat* mat) override;

  PetscErrorCode create_scalar_da();
  virtual PetscErrorCode set_sizes_and_ltog(Mat mat) const = 0;

  DM sda_;
  PetscInt l_size_;
  ISLocalToGlobalMapping v_ltog_, s_ltog_;
};


class Divergence final : public NonRectangularOperator {
public:
  Divergence(DM da);

protected:
  PetscErrorCode set_sizes_and_ltog(Mat mat) const override;
  void fill_stencil(Yee_shift s, PetscInt x, PetscInt y, PetscInt z,
    std::vector<MatStencil>& row, std::vector<MatStencil>& col) const override;
};


class Gradient final : public NonRectangularOperator {
public:
  Gradient(DM da);

private:
  PetscErrorCode set_sizes_and_ltog(Mat mat) const override;
  void fill_stencil(Yee_shift s, PetscInt x, PetscInt y, PetscInt z,
    std::vector<MatStencil>& row, std::vector<MatStencil>& col) const override;
};

#endif  // SRC_UTILS_OPERATORS_H
