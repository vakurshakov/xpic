#ifndef SRC_INTERFACES_COMMAND_H
#define SRC_INTERFACES_COMMAND_H

#include "src/pch.h"

/**
 * @brief We use commands in two situation:
 * 1. To preset simulation before main calculation cycle;
 * 2. To preset simulation on each step of the calculation.
 */
class Command {
 public:
  virtual ~Command() = default;

  /// @param timestep Outer time step, optional.
  virtual void execute(timestep_t timestep) = 0;

  /**
   * @brief Checks whether command needs to be
   * removed from the command list or not.
   *
   * @param timestep Outer time step, base of decision.
   *
   * @return True if command must be removed from the
   * list (false otherwise). By default returns false.
   */
  virtual bool needs_to_be_removed(timestep_t timestep) const { return false; }
};

/// @brief Used to execute command once in the simulation cycle.
class Command_once final : public Command {
 public:
  virtual ~Command_once() = default;

  /// @brief Since command is executed once, it always return true.
  bool needs_to_be_removed(timestep_t timestep) const override { return true; }
};

#endif  // SRC_INTERFACES_COMMAND_H
