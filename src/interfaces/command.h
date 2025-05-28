#ifndef SRC_INTERFACES_COMMAND_H
#define SRC_INTERFACES_COMMAND_H

#include "src/pch.h"
#include "src/utils/utils.h"

namespace interfaces {

/**
 * @brief We use commands in two situation:
 * 1. To preset simulation before main calculation cycle;
 * 2. To preset simulation on each step of the calculation.
 */
class Command {
public:
  Command() = default;
  virtual ~Command() = default;

  /// @brief Explicit finalize should be called to end up the command.
  virtual PetscErrorCode finalize()
  {
    PetscFunctionBeginUser;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /// @param timestep Outer time step, optional.
  virtual PetscErrorCode execute(PetscInt timestep) = 0;
};

}  // namespace interfaces

using Command_up = std::unique_ptr<interfaces::Command>;

#endif  // SRC_INTERFACES_COMMAND_H
