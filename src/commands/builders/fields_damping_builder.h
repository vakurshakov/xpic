#ifndef SRC_COMMANDS_BUILDERS_FIELDS_DAMPING_BUILDER_H
#define SRC_COMMANDS_BUILDERS_FIELDS_DAMPING_BUILDER_H

#include "src/commands/builders/command_builder.h"

class FieldsDampingBuilder : public CommandBuilder {
public:
  FieldsDampingBuilder(
    const interfaces::Simulation& simulation, std::list<Command_up>& result);

  PetscErrorCode build(const Configuration::json_t& info) override;

private:
  std::string_view usage_message() const override
  {
    std::string_view help =
      "{\n"
      "  \"command\": \"FieldsDamping\",\n"
      "  \"E\": \"E\",\n"
      "  \"B\": \"B\",\n"
      "  \"B0\": \"B0\",\n"
      "  \"geometry\": {\n"
      "    \"name\": \"BoxGeometry\",\n"
      "    \"min\": 0,\n"
      "    \"max\": \"Geom\"\n"
      "  }\n"
      "}\n";
    return help;
  }
};

#endif  // SRC_COMMANDS_BUILDERS_FIELDS_DAMPING_BUILDER_H
