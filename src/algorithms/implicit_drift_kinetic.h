#ifndef SRC_ALGORITHMS_IMPLICIT_DRIFT_KINETIC_H
#define SRC_ALGORITHMS_IMPLICIT_DRIFT_KINETIC_H

#include <vector>

#include <petscdm.h>

#include "src/interfaces/point.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/utils/shape.h"
#include "src/utils/geometries.h"
#include "src/impls/drift_kinetic/segments.h"

namespace drift_kinetic {

struct DriftKineticShapeE
{
    static constexpr PetscInt shr = 2;
    static constexpr PetscInt shw = 5;
    Vector3I p_g;
    Vector3R p_s;

    void setup(const DriftKineticSegment& segment);
};

struct DriftKineticShapeGradB
{
    static constexpr PetscInt shr = 2;
    static constexpr PetscInt shw = 5;
    Vector3I p_g;
    Vector3R p_s;

    void setup(const DriftKineticSegment& segment);
};

class DriftKineticEsirkepov{
public:
  DriftKineticEsirkepov(Vector3R*** B_g, Vector3R*** M_g);
  DriftKineticEsirkepov(Vector3R*** B_g);

  DriftKineticEsirkepov(Vector3R*** E_g, Vector3R*** B0_g, Vector3R*** B_g, Vector3R*** J_g, Vector3R*** M_g);

  PetscErrorCode interpolate_B(Vector3R& B_p, const Vector3R& Rp);
  PetscErrorCode interpolate_B0(Vector3R& B0_p, const Vector3R& Rp);

  PetscErrorCode interpolate_E(Vector3R& E_p, const DriftKineticSegment& s);
  PetscErrorCode interpolate_gradB(Vector3R& gradB_p, Vector3R& b_p, const DriftKineticSegment& s);

  PetscErrorCode interpolate(Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p, const Vector3R& Rn, const Vector3R& R0);
  PetscErrorCode interpolate_B(Vector3R& B0_p, Vector3R& meanB_p, Vector3R& Bn_p, const Vector3R& Rn, const Vector3R& R0);

  PetscErrorCode decomposition_M(const Vector3R& Rp, const Vector3R& b_p, PetscReal mu_p);
  PetscErrorCode decomposition_J(const DriftKineticSegment& s,//
  const Vector3R& Vp, PetscReal q_p);

  PetscErrorCode decomposition(const Vector3R& Rn, const Vector3R& R0,//
  const Vector3R& Vp, PetscReal q_p, PetscReal mu_p);

private:

  Vector3R*** E_g = nullptr;

  Vector3R*** B0_g = nullptr;
  Vector3R*** B_g = nullptr;

  Vector3R*** J_g = nullptr;
  Vector3R*** M_g = nullptr;
};

} // namespace drift_kinetic

#endif  // SRC_ALGORITHMS_IMPLICIT_DRIFT_KINETIC_H
