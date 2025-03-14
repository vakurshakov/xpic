#ifndef SRC_ALGORITHMS_ESIRKEPOV_DECOMPOSITION_H
#define SRC_ALGORITHMS_ESIRKEPOV_DECOMPOSITION_H

#include "src/utils/shape.h"

/// @brief Charge-conserving Esirkepov density decomposition.
class EsirkepovDecomposition {
public:
  EsirkepovDecomposition() = delete;
  EsirkepovDecomposition(const Shape& shape, PetscReal alpha);

  /// @brief Decomposition context is a reference to outer global current.
  using Context = Vector3R***;

  /// @note A temporary buffer is allocated with a maximum size of
  /// `shape_width` so that a lower-order shape can fit into it.
  /// @todo Think on std::vector<PetscReal> instead.
  PetscErrorCode process(Context& J) const;

private:
  PetscReal get_jx(PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jx) const;
  PetscReal get_jy(PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jy) const;
  PetscReal get_jz(PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jz) const;

  const Shape& shape;
  PetscReal alpha;
};

#endif  // SRC_ALGORITHMS_ESIRKEPOV_DECOMPOSITION_H
