#ifndef SRC_ALGORITHMS_ESIRKEPOV_DECOMPOSITION_H
#define SRC_ALGORITHMS_ESIRKEPOV_DECOMPOSITION_H

#include "src/utils/particle_shape.h"

/// @brief Charge-conserving Esirkepov density decomposition.
class Esirkepov_decomposition {
public:
  Esirkepov_decomposition(const Vector3I& width, PetscReal alpha,
    const Shape& old_shape, const Shape& new_shape);

  /// @brief Decomposition context is reference to outer global current.
  using Context = Vector3R***;

  void process(const Vector3I& p_g, Context& j_field) const;

private:
  PetscReal get_Jx(PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jx) const;
  PetscReal get_Jy(PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jx) const;
  PetscReal get_Jz(PetscInt x, PetscInt y, PetscInt z, PetscReal* temp_jx) const;

  const Vector3I& width;

  PetscReal alpha;
  const Shape& old_shape;
  const Shape& new_shape;
};

#endif  // SRC_ALGORITHMS_ESIRKEPOV_DECOMPOSITION_H
