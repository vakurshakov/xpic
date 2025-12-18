#ifndef SRC_ALGORITHMS_DRIFT_KINETIC_ESIRKEPOV_H
#define SRC_ALGORITHMS_DRIFT_KINETIC_ESIRKEPOV_H

#include <petscdm.h>

#include "implicit_esirkepov.h"

class DriftKineticEsirkepov{
public:
  DriftKineticEsirkepov(
    Vector3R*** E_g, Vector3R*** B_g, Vector3R*** J_g, Vector3R*** gradB_g);

  DriftKineticEsirkepov(
    Vector3R*** E_g, Vector3R*** B_g, Vector3R*** J_g, Vector3R*** M_g, //
    Vector3R*** dBidx_g, Vector3R*** dBidy_g, Vector3R*** dBidz_g);

  PetscErrorCode set_dBidrj(Vector3R*** _dBidx, Vector3R*** _dBidy, Vector3R*** _dBidz);

  PetscErrorCode interpolate(Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p,
    const Vector3R& rn, const Vector3R& r0);

  PetscErrorCode interpolate_E(Vector3R& E_p, const Vector3R& rn, const Vector3R& r0);
  PetscErrorCode interpolate_B(Vector3R& B_p, const Vector3R& rn);
  PetscErrorCode interpolate_gradB(Vector3R& gradB_p, Vector3R& b_p, //
  const Vector3R& rn, const Vector3R& r0);

  PetscErrorCode decomposition(const Vector3R& Rsn, //
  const Vector3R& Rs0, const Vector3R& Vp, PetscReal q_p,  PetscReal mu_p);

  PetscErrorCode decomposition_J(const Vector3R& Rsn, const Vector3R& Rs0,//
  const Vector3R& Vp, PetscReal q_p);
  PetscErrorCode decomposition_M(const Vector3R& Rsn, PetscReal mu_p);

private:
  Vector3R*** dBidx_g = nullptr;
  Vector3R*** dBidy_g = nullptr;
  Vector3R*** dBidz_g = nullptr;

  Vector3R*** E_g;
  Vector3R*** B_g;

  Vector3R*** J_g;
  Vector3R*** M_g;

  Vector3R*** gradB_g = nullptr;
};

#endif  // SRC_ALGORITHMS_DRIFT_KINETIC_ESIRKEPOV_H
