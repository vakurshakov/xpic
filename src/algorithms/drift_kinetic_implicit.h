#ifndef SRC_ALGORITHMS_DRIFT_KINETIC_ESIRKEPOV_H
#define SRC_ALGORITHMS_DRIFT_KINETIC_ESIRKEPOV_H

#include <petscdm.h>

#include "implicit_esirkepov.h"

class DriftKineticEsirkepov{
public:
  DriftKineticEsirkepov(
    Vector3R*** E_g, Vector3R*** B_g, Vector3R*** J_g, Vector3R*** gradB_g);

  PetscErrorCode set_dBidrj(Vector3R*** dBdx, Vector3R*** dBdy, Vector3R*** dBdz);

  PetscErrorCode interpolate(Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p,
    const Vector3R& rn, const Vector3R& r0);

  PetscErrorCode interpolate_E(Vector3R& E_p, const Vector3R& rn, const Vector3R& r0);
  PetscErrorCode interpolate_B(Vector3R& B_p, const Vector3R& rn);
  PetscErrorCode interpolate_gradB(Vector3R& gradB_p, Vector3R& b_p, const Vector3R& rn, const Vector3R& r0);


private:
  Vector3R*** dBdx_g = nullptr;
  Vector3R*** dBdy_g = nullptr;
  Vector3R*** dBdz_g = nullptr;

  Vector3R*** E_g;
  Vector3R*** B_g;
  Vector3R*** J_g;

  Vector3R*** gradB_g;
};

#endif  // SRC_ALGORITHMS_DRIFT_KINETIC_ESIRKEPOV_H
