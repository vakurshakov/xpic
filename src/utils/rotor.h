#ifndef SRC_VECTORS_OPERATORS_H
#define SRC_VECTORS_OPERATORS_H

#include <petscdmda.h>
#include <petscmat.h>

#include "src/utils/utils.h"

class Rotor {
public:
  Rotor(DM da);

  PetscErrorCode set_positive(Mat mat);
  PetscErrorCode set_negative(Mat mat);

private:
  /// @note in natural ordering, debug purpose
  PetscInt index(PetscInt x, PetscInt y, PetscInt z, PetscInt c);

  const PetscReal values_x[4];
  const PetscReal values_y[4];
  const PetscReal values_z[4];

  DM da_;
  PetscInt start_[3], size_[3];
};

#endif // SRC_VECTORS_OPERATORS_H
