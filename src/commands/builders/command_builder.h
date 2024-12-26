#ifndef SRC_BUILDERS_COMMAND_BUILDER_H
#define SRC_BUILDERS_COMMAND_BUILDER_H

#include "src/pch.h"
#include "src/interfaces/builder.h"
#include "src/interfaces/command.h"
#include "src/impls/basic/simulation.h"
#include "src/utils/configuration.h"

class CommandBuilder : public interfaces::Builder {
public:
  CommandBuilder(
    const interfaces::Simulation& simulation, std::list<Command_up>& result);

protected:
  using Commands_list = std::list<Command_up>;
  Commands_list& commands_;
};


PetscErrorCode build_commands(const interfaces::Simulation& simulation,
  std::string_view name, std::list<Command_up>& result);

#endif  // SRC_BUILDERS_COMMAND_BUILDER_H
