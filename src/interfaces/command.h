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
  DEFAULT_MOVABLE(Command);

  Command() = default;
  virtual ~Command() = default;

  /// @param timestep Outer time step, optional.
  virtual PetscErrorCode execute(PetscInt timestep) = 0;

  /**
   * @brief Checks whether command needs to be
   * removed from the command list or not.
   *
   * @param timestep Outer time step, base of decision.
   *
   * @return True if command must be removed from the
   * list (false otherwise). By default returns false.
   */
  virtual bool needs_to_be_removed(PetscInt /* timestep */) const
  {
    return false;
  }
};

}  // namespace interfaces

using Command_up = std::unique_ptr<interfaces::Command>;

#endif  // SRC_INTERFACES_COMMAND_H
