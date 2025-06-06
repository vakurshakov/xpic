#ifndef SRC_COMMANDS_SET_MAGNETIC_FIELD_H
#define SRC_COMMANDS_SET_MAGNETIC_FIELD_H

#include <petscvec.h>

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/utils/vector3.h"

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
