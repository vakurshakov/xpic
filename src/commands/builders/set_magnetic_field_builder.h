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
      "\nStructure of the SetMagneticField / SetElectricField command description:\n"
      "{\n"
      "  \"command\": \"SetMagneticField\" | \"SetElectricField\", -- Name of the command.\n"
      "  \"setter\": { -- Field setter description.\n"
      "    \"name\": \"Name\", -- Setter name, one of the following:\n"
      "                        SetUniformField, SetCoilsField, SetCosineField.\n"
      "    Specific description of a setter chosen by \"Name\"...\n"
      "  }\n"
      "}\n";
    return help;
  }
};

#endif  // SRC_COMMANDS_BUILDERS_SET_MAGNETIC_FIELD_BUILDER_H
