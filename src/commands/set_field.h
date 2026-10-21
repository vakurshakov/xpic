#ifndef SRC_COMMANDS_SET_FIELD_H
#define SRC_COMMANDS_SET_FIELD_H

#include <petscvec.h>

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/utils/vector3.h"


/// @brief Component `c` of a Yee-staggered field lives at the point
/// `(idx[a] + offset[c][a]) * d[a]` along each axis `a`.
struct Stagger {
  PetscReal offset[3][3];
};

constexpr Stagger yee_E{{
  {0.5, 0.0, 0.0},
  {0.0, 0.5, 0.0},
  {0.0, 0.0, 0.5},
}};

constexpr Stagger yee_B{{
  {0.0, 0.5, 0.5},
  {0.5, 0.0, 0.5},
  {0.5, 0.5, 0.0},
}};

/// @brief Fills `field` with `setup`; if `field0` is given, adds the result to it.
/// @todo Simulation should be passed
class SetField : public interfaces::Command {
public:
  using Setter = std::function<PetscErrorCode(Vec, const Stagger&)>;
  SetField(Vec field, Vec field0, const Stagger& stagger, Setter&& setup);
  PetscErrorCode execute(PetscInt t) override;

private:
  Vec field = NULL;
  Vec field0 = NULL;
  const Stagger& stagger;
  Setter setup;
};

struct SetUniformField {
  PetscErrorCode operator()(Vec vec, const Stagger& stagger);
  Vector3R value;
};

/**
 * @brief Linearly varying field F(r) = value * (1 + (r[axis] - origin) /
 * length). The direction stays fixed (straight field lines); only the magnitude
 * ramps along `axis`, giving a constant gradient |value| / length. `origin`
 * shifts the reference point at which the field equals `value` (default 0).
 */
struct SetGradientField {
  PetscErrorCode operator()(Vec vec, const Stagger& stagger);
  Vector3R value;
  PetscInt axis;
  PetscReal length;
  PetscReal origin = 0.0;
};

/**
 * @brief Rotational field with circular field lines
 * around an axis parallel to z through (center_x, center_y).
 *  - radius <= 0: B = value * e_phi = value (-(y-cy), x-cx, 0)/r, i.e. constant
 *    magnitude |B| = value (no gradient drift, but ~1/r and singular at r=0).
 *  - radius  > 0: B = value (-(y-cy), x-cx, 0)/radius, i.e. a *linear* field
 *    (exact rot B, no singularity); |B| = value * r/radius equals `value` on
 *    the circle r = radius. Combined with v_perp = 0 (mu = 0) the gradient
 *    drift vanishes, so this isolates the curvature drift with a near-linear
 *    field.
 */
struct SetAzimuthalField {
  PetscErrorCode operator()(Vec vec, const Stagger& stagger);
  PetscReal value;
  PetscReal center_x;
  PetscReal center_y;
  PetscReal radius = 0.0;
};

struct SetCosineField {
  PetscErrorCode operator()(Vec vec, const Stagger& stagger);

  Vector3R amplitude;
  Vector3R wave_number;
  Vector3R phase{};
};

struct SetGeneralCosineField {
  PetscErrorCode operator()(Vec vec, const Stagger& stagger);

  Vector3R amplitude;
  Vector3R wave_number;
};

struct SetCoilsField {
  struct Coil {
    PetscReal z0;
    PetscReal R;
    PetscReal I;
  };
  std::vector<Coil> coils;

  SetCoilsField(std::vector<Coil>&& coils);
  PetscErrorCode operator()(Vec vec, const Stagger& stagger);

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

#endif  // SRC_COMMANDS_SET_FIELD_H
