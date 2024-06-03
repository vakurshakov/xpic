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
  bool remap_positive(PetscInt& x, PetscInt& y, PetscInt& z);
  bool remap_negative(PetscInt& x, PetscInt& y, PetscInt& z);

  PetscInt index(PetscInt k, PetscInt j, PetscInt i, PetscInt l);

  DM da_;
  DMBoundaryType bounds_[3];
  PetscInt start_[3], size_[3];
};

#endif // SRC_VECTORS_OPERATORS_H
