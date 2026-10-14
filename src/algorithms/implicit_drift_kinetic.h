#ifndef SRC_ALGORITHMS_IMPLICIT_DRIFT_KINETIC_H
#define SRC_ALGORITHMS_IMPLICIT_DRIFT_KINETIC_H

/// @brief Drift-kinetic field interpolation and deposition.

#include "src/impls/drift_kinetic/segments.h"

namespace drift_kinetic {

/// @brief Grid stencil for electric fields and current.
struct DriftKineticShapeE {
  static constexpr PetscInt shr = 1;
  static constexpr PetscInt shw = 3;
  Vector3I p_g;

  void setup(const DriftKineticSegment& segment);
};

/// @brief Grid stencil for magnetic field derivatives.
struct DriftKineticShapeGradB {
  static constexpr PetscInt shr = 2;
  static constexpr PetscInt shw = 4;
  Vector3I p_g;
  Vector3R p_s;

  void setup(const DriftKineticSegment& segment);
};

/// @brief Interpolates fields and deposits current and magnetization.
class DriftKineticEsirkepov {
public:
  /// @brief Binds a magnetic field for point interpolation.
  DriftKineticEsirkepov(Vector3R*** B_g);

  /// @brief Binds field layers and deposition arrays.
  DriftKineticEsirkepov(Vector3R*** E_g, Vector3R*** Bn0_g, Vector3R*** Bnh_g,
    Vector3R*** Bn1_g, Vector3R*** J_g, Vector3R*** M_g);

  PetscErrorCode interpolate(Vector3R& E_p, PetscReal& lenB_p, Vector3R& b_p,
    Vector3R& gradB_p, Vector3R& rotB_p, const Vector3R& Rn, const Vector3R& R0);

  PetscErrorCode interpolate_B(Vector3R& B_p, const Vector3R& Rp);

  PetscErrorCode decomposition(const Vector3R& Rn, const Vector3R& R0,
    const Vector3R& Vp, PetscReal q_p, PetscReal mu_p);

private:
  PetscErrorCode interpolate_B(  //
    Vector3R& B_p, Vector3R*** B_g, const Vector3R& Rp);

  PetscErrorCode interpolate_E(  //
    Vector3R& E_p, const DriftKineticSegment& s);

  PetscErrorCode interpolate_rotB(  //
    Vector3R& rotB_p, const DriftKineticSegment& s);

  PetscErrorCode interpolate_gradB(  //
    Vector3R& gradB_p, const Vector3R& b_p, const DriftKineticSegment& s);

  PetscErrorCode decomposition_M(  //
    const Vector3R& Rp, const Vector3R& b_p, PetscReal mu_p);

  PetscErrorCode decomposition_J(  //
    const DriftKineticSegment& s, const Vector3R& Vp, PetscReal q_p);

  Vector3R*** E_g = nullptr;  // Midpoint electric field.

  Vector3R*** Bn0_g = nullptr;  // Previous magnetic field.
  Vector3R*** Bnh_g = nullptr;  // Midpoint magnetic field.
  Vector3R*** Bn1_g = nullptr;  // Next magnetic field.

  Vector3R*** J_g = nullptr;  // Current deposition array.
  Vector3R*** M_g = nullptr;  // Magnetization deposition array.
};

}  // namespace drift_kinetic

#endif  // SRC_ALGORITHMS_IMPLICIT_DRIFT_KINETIC_H
