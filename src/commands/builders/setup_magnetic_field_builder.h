#ifndef SRC_COMMANDS_BUILDERS_SETUP_MAGNETIC_FIELD_BUILDER_H
#define SRC_COMMANDS_BUILDERS_SETUP_MAGNETIC_FIELD_BUILDER_H

#include "src/commands/builders/command_builder.h"


class SetupMagneticFieldBuilder : public CommandBuilder {
public:
  SetupMagneticFieldBuilder(
    const interfaces::Simulation& simulation, std::list<Command_up>& result);

  PetscErrorCode build(const Configuration::json_t& json) override;

private:
  std::string_view usage_message() const override
  {
    return "";
  }
};

#endif  // SRC_COMMANDS_BUILDERS_SETUP_MAGNETIC_FIELD_BUILDER_H
