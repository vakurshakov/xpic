#ifndef SRC_ALGORITHMS_IMPLICIT_DRIFT_KINETIC_H
#define SRC_ALGORITHMS_IMPLICIT_DRIFT_KINETIC_H

#include <vector>

#include <petscdm.h>

#include "src/interfaces/point.h"
#include "src/algorithms/simple_interpolation.h"
#include "src/utils/shape.h"
#include "src/utils/geometries.h"
#include "src/impls/drift_kinetic/segments.h"

/// @file implicit_drift_kinetic.h
/// @brief Drift-kinetic field interpolation and deposition.

namespace drift_kinetic {

/// @brief Grid stencil for electric fields and current.
struct DriftKineticShapeE
{
    static constexpr PetscInt shr = 1;
    static constexpr PetscInt shw = 3;
    Vector3I p_g;

    /// @brief Sets the stencil origin for a segment.
    void setup(const DriftKineticSegment& segment);
};

/// @brief Grid stencil for magnetic field derivatives.
struct DriftKineticShapeGradB
{
    static constexpr PetscInt shr = 2;
    static constexpr PetscInt shw = 4;
    Vector3I p_g;
    Vector3R p_s;

    /// @brief Sets the stencil origin and splitting coordinate.
    void setup(const DriftKineticSegment& segment);
};

/// @brief Interpolates fields and deposits current and magnetization.
class DriftKineticEsirkepov{
public:
  /// @brief Binds a magnetic field for point interpolation.
  DriftKineticEsirkepov(Vector3R*** B_g);

  /// @brief Interpolates the latest bound magnetic field at a point.
  PetscErrorCode interpolate_B(Vector3R& B_p, const Vector3R& Rp);

  /// @brief Binds field layers and deposition arrays.
  DriftKineticEsirkepov(Vector3R*** E_g, Vector3R*** Bn_g, Vector3R*** Bnh_g, //
                        Vector3R*** Bn1_g, Vector3R*** J_g, Vector3R*** M_g);

  /// @brief Interpolates fields and magnetic derivatives along a track.
  PetscErrorCode interpolate(Vector3R& E_p, PetscReal& lenB_p, Vector3R& b_p, Vector3R& gradB_p, //
                             Vector3R& rotB_p, const Vector3R& Rn, const Vector3R& R0);

  /// @brief Deposits current and magnetization for a track.
  PetscErrorCode decomposition(const Vector3R& Rn, const Vector3R& R0,//
                               const Vector3R& Vp, PetscReal q_p, PetscReal mu_p);

private:
  /// @brief Interpolates the electric field along a segment.
  PetscErrorCode interpolate_E(Vector3R& E_p, const DriftKineticSegment& s);
  /// @brief Interpolates the magnetic magnitude gradient along a segment.
  PetscErrorCode interpolate_gradB(Vector3R& gradB_p, const Vector3R& b_p, const DriftKineticSegment& s);
  /// @brief Interpolates the magnetic field curl along a segment.
  PetscErrorCode interpolate_rotB(Vector3R& rotB_p, const DriftKineticSegment& s);
  /// @brief Interpolates a selected magnetic field at a point.
  PetscErrorCode interpolate_B(Vector3R& B_p, Vector3R*** B_g, const Vector3R& Rp);
  /// @brief Deposits magnetization at a track endpoint.
  PetscErrorCode decomposition_M(const Vector3R& Rp, const Vector3R& b_p, PetscReal mu_p);
  /// @brief Deposits current along a segment.
  PetscErrorCode decomposition_J(const DriftKineticSegment& s, const Vector3R& Vp, PetscReal q_p);

  Vector3R*** E_g = nullptr;  ///< Midpoint electric field.

  Vector3R*** Bn_g = nullptr;   ///< Previous magnetic field.
  Vector3R*** Bnh_g = nullptr;  ///< Midpoint magnetic field.
  Vector3R*** Bn1_g = nullptr;  ///< Next magnetic field.

  Vector3R*** J_g = nullptr;  ///< Current deposition array.
  Vector3R*** M_g = nullptr;  ///< Magnetization deposition array.
};

} // namespace drift_kinetic

#endif  // SRC_ALGORITHMS_IMPLICIT_DRIFT_KINETIC_H
