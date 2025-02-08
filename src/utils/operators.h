#ifndef SRC_UTILS_OPERATORS_H
#define SRC_UTILS_OPERATORS_H

#include <petscdmda.h>
#include <petscis.h>
#include <petscmat.h>

#include "src/utils/utils.h"
#include "src/utils/vector3.h"

/// @brief Utility class to create constant operators on a `DMDA` grid.
class Operator {
public:
  DEFAULT_MOVABLE(Operator);

  Operator() = default;
  virtual ~Operator() = default;

  /// @note `idxm` will be modified to be an array of local indices!
  static PetscErrorCode remap_stencil(
    DM da, PetscInt mdof, PetscInt m, PetscInt* idxm);

protected:
  Operator(DM da, PetscInt mdof = 3, PetscInt ndof = 3);

  PetscInt m_index(PetscInt z, PetscInt y, PetscInt x, PetscInt c) const;
  PetscInt n_index(PetscInt z, PetscInt y, PetscInt x, PetscInt c) const;

  DM da_;
  Vector3I start_, size_;
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

  enum class Yee_shift : std::uint8_t {
    Positive,
    Negative,
  };

  virtual PetscErrorCode create_matrix(Mat* mat);

  /// @brief Remaps the values from `MatStencil` into `PetscInt` arrays and then
  /// passes it to default MatSetPreallocationCOO(), (different dof can be used).
  PetscErrorCode fill_matrix(Mat mat, Yee_shift sh);

  /// @brief Specifies the stencil for each point `(z, y, x)` in space, after
  /// that whole chunk of `values` will be inserted into the matrix at once.
  virtual void fill_stencil(Yee_shift sh, PetscInt z, PetscInt y, PetscInt x,
    MatStencil* coo_i, MatStencil* coo_j) const = 0;

  inline PetscInt ind(PetscInt c, PetscInt i) const;

  const std::vector<PetscReal> values_;
};

class Rotor final : public FiniteDifferenceOperator {
public:
  Rotor(DM da);

private:
  void fill_stencil(Yee_shift sh, PetscInt z, PetscInt y, PetscInt x,
    MatStencil* coo_i, MatStencil* coo_j) const override;
};

class RotorMult final : public FiniteDifferenceOperator {
public:
  RotorMult(DM da);

  /// @note Implements only `negative * positive` product
  PetscErrorCode create(Mat* mat);

private:
  void fill_stencil(Yee_shift, PetscInt z, PetscInt y, PetscInt x,
    MatStencil* coo_i, MatStencil* coo_j) const override;

  using FiniteDifferenceOperator::create_negative;
  using FiniteDifferenceOperator::create_positive;
};

class NonRectangularOperator : public FiniteDifferenceOperator {
public:
  DEFAULT_MOVABLE(NonRectangularOperator);
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
  void fill_stencil(Yee_shift sh, PetscInt z, PetscInt y, PetscInt x,
    MatStencil* coo_i, MatStencil* coo_j) const override;
};

class Gradient final : public NonRectangularOperator {
public:
  Gradient(DM da);

private:
  PetscErrorCode set_sizes_and_ltog(Mat mat) const override;
  void fill_stencil(Yee_shift sh, PetscInt z, PetscInt y, PetscInt x,
    MatStencil* coo_i, MatStencil* coo_j) const override;
};

#endif  // SRC_UTILS_OPERATORS_H
