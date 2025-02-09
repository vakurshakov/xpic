#ifndef SRC_COMMANDS_SET_MAGNETIC_FIELD_H
#define SRC_COMMANDS_SET_MAGNETIC_FIELD_H

#include <petscvec.h>

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/utils/vector3.h"

class SetMagneticField : public interfaces::Command {
public:
  using Setter = std::function<PetscErrorCode(Vec /* storage */)>;
  SetMagneticField(Vec storage, Setter&& setup);

  PetscErrorCode execute(timestep_t t) override;

private:
  Vec storage_;
  Setter setup_;
};

class SetUniformField {
public:
  SetUniformField(const Vector3R& value);
  PetscErrorCode operator()(Vec storage);
  Vector3R value_;
};

#endif // SRC_COMMANDS_SET_MAGNETIC_FIELD_H
