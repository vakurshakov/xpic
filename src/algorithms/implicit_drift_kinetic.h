#ifndef SRC_ALGORITHMS_IMPLICIT_DRIFT_KINETIC_H
#define SRC_ALGORITHMS_IMPLICIT_DRIFT_KINETIC_H

/// @brief Drift-kinetic field interpolation and deposition.

#include "src/impls/drift_kinetic/segments.h"

/// @brief 1 builds the segment weights from one-dimensional spline tables
/// (`DriftKineticSplineTables`), 0 evaluates the splines at every stencil node.
/// Both give bitwise identical results.
#define DRIFT_KINETIC_SPLINE_TABLES 1

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

#if DRIFT_KINETIC_SPLINE_TABLES
/// @brief Separable factors of the segment weights on the 4-node stencil of
/// `DriftKineticShapeGradB`; every weight is a product of two entries.
struct DriftKineticSplineTables {
  static constexpr PetscInt shw = DriftKineticShapeGradB::shw;

  PetscReal along_ii[3][shw];  // S1 along axis a, for S_aa and E, J.
  PetscReal along_ij[3][shw];  // S1 along axis a, split at the cell node.

  // acrossS2 over the axes b = a + 1, c = a + 2 (cyclic), indexed by the
  // half-step shifts m = 2 * shift_b + shift_c: 0 for E and J, 1 for S_ab,
  // 2 for S_ac, 3 for S_aa.
  PetscReal across[3][4][shw][shw];

  void setup(const DriftKineticSegment& segment);
};
#endif

/// @brief Interpolates fields and deposits current and magnetization.
class DriftKineticEsirkepov {
public:
  /// @brief Binds a magnetic field for point interpolation.
  DriftKineticEsirkepov(Vector3R*** B_g);

  /// @brief Binds field layers and deposition arrays.
  DriftKineticEsirkepov(Vector3R*** E_g, Vector3R*** Bn0_g, Vector3R*** Bnh_g,
    Vector3R*** Bn1_g, Vector3R*** J_g, Vector3R*** M_g);

  /// @brief Selects the time interval within the field step.
  void set_interval(PetscReal begin, PetscReal end);
  void set_grid_bounds(const Vector3I& begin, const Vector3I& end);

  PetscErrorCode interpolate(Vector3R& E_p, PetscReal& lenB_p, Vector3R& b_p,
    Vector3R& gradB_p, Vector3R& rotB_p, const Vector3R& Rn, const Vector3R& R0);

  PetscErrorCode decomposition(const Vector3R& Rn, const Vector3R& R0,
    const Vector3R& Vh, PetscReal q_p, PetscReal mu_p);

  PetscErrorCode interpolate_B(Vector3R& B_p, const Vector3R& Rp);

private:
  Vector3R magnetic(PetscInt x, PetscInt y, PetscInt z, PetscReal theta) const;
  PetscErrorCode interpolate_B_at(Vector3R& B_p, const Vector3R& Rp, PetscReal theta);
  PetscErrorCode check_track(const Vector3R& R0, const Vector3R& Rn) const;
  PetscErrorCode check_stencil(const Vector3I& begin, const Vector3I& end) const;

  PetscReal theta_begin = 0, theta_end = 1;
  bool bounded = false;
  Vector3I grid_begin, grid_end;

  PetscErrorCode interpolate_B(  //
    Vector3R& B_p, Vector3R*** B_g, const Vector3R& Rp);

  PetscErrorCode interpolate_E(  //
    Vector3R& E_p, const DriftKineticSegment& s);

#if DRIFT_KINETIC_SPLINE_TABLES
  PetscErrorCode interpolate_gradB_rotB(Vector3R& gradB_p, Vector3R& rotB_p,
    const Vector3R& b_p, const DriftKineticSegment& s);
#else
  PetscErrorCode interpolate_rotB(  //
    Vector3R& rotB_p, const DriftKineticSegment& s);

  PetscErrorCode interpolate_gradB(  //
    Vector3R& gradB_p, const Vector3R& b_p, const DriftKineticSegment& s);
#endif

  PetscErrorCode decomposition_M(  //
    const Vector3R& Rp, const Vector3R& b_p, PetscReal mu_p);

  PetscErrorCode decomposition_J(  //
    const DriftKineticSegment& s, const Vector3R& Vh, PetscReal q_p);

  Vector3R*** E_g = nullptr;  // Midpoint electric field.

  Vector3R*** Bn0_g = nullptr;  // Previous magnetic field.
  Vector3R*** Bnh_g = nullptr;  // Midpoint magnetic field.
  Vector3R*** Bn1_g = nullptr;  // Next magnetic field.

  Vector3R*** J_g = nullptr;  // Current deposition array.
  Vector3R*** M_g = nullptr;  // Magnetization deposition array.

#if DRIFT_KINETIC_SPLINE_TABLES
  DriftKineticSplineTables tables;  // Weights of the current segment.
#endif
};

}  // namespace drift_kinetic

#endif  // SRC_ALGORITHMS_IMPLICIT_DRIFT_KINETIC_H
