#ifndef SRC_COMMANDS_SET_MAGNETIC_FIELD_H
#define SRC_COMMANDS_SET_MAGNETIC_FIELD_H

#include <petscvec.h>

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/utils/geometries.h"
#include "src/utils/vector3.h"

/// @todo Simulation should be passed

class SetMagneticField : public interfaces::Command {
public:
  using Setter = std::function<PetscErrorCode(Vec)>;
  SetMagneticField(Vec B0, Vec B, Setter&& setup);
  PetscErrorCode execute(PetscInt t) override;

private:
  Vec B0_ = nullptr;
  Vec B_ = nullptr;
  Setter setup_;
};

struct SetUniformField {
  SetUniformField(const Vector3R& value);
  PetscErrorCode operator()(Vec vec);
  Vector3R value_;
};

/// @brief Linearly varying field F(r) = value * (1 + (r[axis] - origin) / length).
/// The direction stays fixed (straight field lines); only the magnitude ramps
/// along `axis`, giving a constant gradient |value| / length. `origin` shifts the
/// reference point at which the field equals `value` (default 0).
struct SetGradientField {
  SetGradientField(
    const Vector3R& value, PetscInt axis, PetscReal length, PetscReal origin = 0.0);
  PetscErrorCode operator()(Vec vec);
  Vector3R value_;
  PetscInt axis_;
  PetscReal length_;
  PetscReal origin_;
};

/// @brief Rotational field with circular field lines around an axis parallel to
/// z through (center_x, center_y).
///   - radius <= 0: B = value * e_phi = value (-(y-cy), x-cx, 0)/r, i.e. constant
///     magnitude |B| = value (no gradient drift, but ~1/r and singular at r=0).
///   - radius  > 0: B = value (-(y-cy), x-cx, 0)/radius, i.e. a *linear* field
///     (exact rot B, no singularity); |B| = value * r/radius equals `value` on
///     the circle r = radius. Combined with v_perp = 0 (mu = 0) the gradient
///     drift vanishes, so this isolates the curvature drift with a near-linear
///     field.
struct SetAzimuthalField {
  SetAzimuthalField(
    PetscReal value, PetscReal center_x, PetscReal center_y, PetscReal radius = 0.0);
  PetscErrorCode operator()(Vec vec);
  PetscReal value_;
  PetscReal center_x_;
  PetscReal center_y_;
  PetscReal radius_;
};

struct SetCosineField {
  SetCosineField(
    BoxGeometry box, const Vector3R& amplitude, const Vector3R& wave_number);
  PetscErrorCode operator()(Vec vec);

  BoxGeometry box;
  Vector3R amplitude;
  Vector3R wave_number;
};

struct SetGeneralCosineField {
  SetGeneralCosineField(
    BoxGeometry box, const Vector3R& amplitude, const Vector3R& wave_number);
  PetscErrorCode operator()(Vec vec);

  BoxGeometry box;
  Vector3R amplitude;
  Vector3R wave_number;
};

struct SetCoilsField {
  struct Coil {
    PetscReal z0;
    PetscReal R;
    PetscReal I;
  };
  std::vector<Coil> coils_;

  SetCoilsField(std::vector<Coil>&& coils);
  PetscErrorCode operator()(Vec vec);

private:
  static constexpr PetscReal denominator_tolerance = 1e-10;

  static constexpr PetscInt N = 2000;
  static constexpr PetscReal hp = 2 * M_PI / N;
  PetscReal cos[N];

  PetscReal get_Br(PetscReal z, PetscReal r);
  PetscReal get_Bz(PetscReal z, PetscReal r);
  PetscReal get_integ_r(PetscReal z, PetscReal r, PetscReal R);
  PetscReal get_integ_z(PetscReal z, PetscReal r, PetscReal R);
};

struct SetApproximateMirrorField {
  PetscReal D;
  PetscReal R;
  PetscReal I;

  SetApproximateMirrorField(PetscReal D, PetscReal R, PetscReal I);
  PetscErrorCode operator()(Vec vec);

private:
  PetscReal get_B0(PetscReal z, PetscReal sign);
  PetscReal get_B1(PetscReal z, PetscReal sign);
};

#endif // SRC_COMMANDS_SET_MAGNETIC_FIELD_H
