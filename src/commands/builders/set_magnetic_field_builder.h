#ifndef SRC_COMMANDS_BUILDERS_SET_MAGNETIC_FIELD_BUILDER_H
#define SRC_COMMANDS_BUILDERS_SET_MAGNETIC_FIELD_BUILDER_H

#include "src/commands/builders/command_builder.h"

class SetMagneticFieldBuilder : public CommandBuilder {
public:
  SetMagneticFieldBuilder(
    interfaces::Simulation& simulation, std::vector<Command_up>& result);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the SetMagneticField command description:\n"
      "{\n"
      "  \"command\": \"SetMagneticField\", -- Name of the command, constant.\n"
      "  \"field\": \"B0\", -- Name of the field to be set (from Simulation).\n"
      "  \"setter\": { -- Magnetic field setter description.\n"
      "    \"name\": \"Name\", -- Setter name, one of the following:\n"
      "                        SetUniformField, SetCoilsField.\n"
      "    Specific description of a setter chosen by \"Name\"...\n"
      "  }\n"
      "}\n";
    return help;
  }
};

#endif  // SRC_COMMANDS_BUILDERS_SET_MAGNETIC_FIELD_BUILDER_H
