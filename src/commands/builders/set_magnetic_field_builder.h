#ifndef SRC_COMMANDS_BUILDERS_SET_MAGNETIC_FIELD_BUILDER_H
#define SRC_COMMANDS_BUILDERS_SET_MAGNETIC_FIELD_BUILDER_H

#include "src/commands/builders/command_builder.h"

class SetMagneticFieldBuilder : public CommandBuilder {
public:
  SetMagneticFieldBuilder(
    const interfaces::Simulation& simulation, std::list<Command_up>& result);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "{\n"
      "  \"command\": \"SetMagneticField\",\n"
      "  \"field\": \"B0\",\n"
      "  \"value\": [0, 0, 0.2]\n"
      "}\n";
    return help;
  }
};

#endif  // SRC_COMMANDS_BUILDERS_SET_MAGNETIC_FIELD_BUILDER_H
