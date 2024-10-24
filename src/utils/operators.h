#ifndef SRC_UTILS_OPERATORS_H
#define SRC_UTILS_OPERATORS_H

#include <petscdmda.h>
#include <petscmat.h>

#include "src/utils/utils.h"

/// @brief Utility to create constant operators on a `DMDA` grid, acts as a storage of da information.
class Operator {
protected:
  Operator(DM da);

  /// @note in natural ordering, debug purpose
  PetscInt index(PetscInt x, PetscInt y, PetscInt z, PetscInt c) const;

  DM da_;
  PetscInt start_[3], size_[3];
};


class Identity : public Operator {
public:
  Identity(DM da);
  PetscErrorCode create(Mat* mat) const;
};


/**
 * @brief This structure serves as an abstraction of derivatives in finite-difference approximation.
 * Yee stencil is used, so each operator can be represented with both positive/negative offsets.
 */
class Finite_difference_operator : public Operator {
public:
  PetscErrorCode create_positive(Mat* mat) const;
  PetscErrorCode create_negative(Mat* mat) const;

protected:
  /// @brief Can not be created explicitly as it is abstract operator.
  Finite_difference_operator(DM da, const std::vector<PetscReal>& values);

  enum class Shift {
    Positive,
    Negative,
  };

  virtual PetscErrorCode create_matrix(Mat* mat) const;

  PetscErrorCode fill_matrix(Mat mat, Shift s) const;

  /**
   * @brief Inherited instances should specify the stencil for each point `(x, y, z)` in space,
   * so after that whole chunk of `values` will be inserted into the matrix at once.
   */
  virtual void fill_stencil(Shift s, PetscInt x, PetscInt y, PetscInt z,
    std::vector<MatStencil>& row, std::vector<MatStencil>& col) const = 0;

  std::tuple<REP3(PetscInt)> get_positive_offsets(PetscInt x, PetscInt y, PetscInt z) const;
  std::tuple<REP3(PetscInt)> get_negative_offsets(PetscInt x, PetscInt y, PetscInt z) const;

  const std::vector<PetscReal> values;
};


class Rotor final : public Finite_difference_operator {
public:
  Rotor(DM da);

private:
  void fill_stencil(Shift s, PetscInt x, PetscInt y, PetscInt z,
    std::vector<MatStencil>& row, std::vector<MatStencil>& col) const override;
};


class Divergence : public Finite_difference_operator {
public:
  Divergence(DM da);

protected:
  PetscErrorCode create_matrix(Mat* mat) const final;
  virtual PetscErrorCode set_sizes(Mat mat) const;

  void fill_stencil(Shift s, PetscInt x, PetscInt y, PetscInt z,
    std::vector<MatStencil>& row, std::vector<MatStencil>& col) const override;
};


class Gradient : public Divergence {
public:
  Gradient(DM da);

private:
  PetscErrorCode set_sizes(Mat mat) const override;

  void fill_stencil(Shift s, PetscInt x, PetscInt y, PetscInt z,
    std::vector<MatStencil>& row, std::vector<MatStencil>& col) const override;
};

#endif // SRC_UTILS_OPERATORS_H
