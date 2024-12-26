#ifndef SRC_COMMANDS_SETUP_MAGNETIC_FIELD_H
#define SRC_COMMANDS_SETUP_MAGNETIC_FIELD_H

#include <petscvec.h>

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/utils/vector3.h"

class SetupMagneticField : public interfaces::Command {
public:
  SetupMagneticField(Vec storage, const Vector3R& value);

  PetscErrorCode execute(timestep_t t) override;

private:
  Vec storage_;

  using Setter = std::function<PetscErrorCode(Vec /* storage */)>;
  Setter setup_;

  class UniformField;
};

#endif // SRC_COMMANDS_SETUP_MAGNETIC_FIELD_H
