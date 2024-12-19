#ifndef SRC_COMMANDS_SETUP_MAGNETIC_FIELD_H
#define SRC_COMMANDS_SETUP_MAGNETIC_FIELD_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/particles.h"


class SetupMagneticField : public interfaces::Command {
public:
  SetupMagneticField(DM da, Vec storage, const Vector3R& value);

  PetscErrorCode execute(timestep_t t) override;

private:
  DM da_;
  Vec storage_;
  Vector3R value_;
};

#endif // SRC_COMMANDS_SETUP_MAGNETIC_FIELD_H
