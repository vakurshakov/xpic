#ifndef SRC_COMMANDS_BUILDERS_REMOVE_PARTICLES_BUILDER_H
#define SRC_COMMANDS_BUILDERS_REMOVE_PARTICLES_BUILDER_H

#include "src/commands/builders/particles_builder.h"
#include "src/utils/particles_load.h"

class RemoveParticlesBuilder : public interfaces::ParticlesBuilder {
public:
  RemoveParticlesBuilder(
    interfaces::Simulation& simulation, std::vector<Command_up>& result);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the RemoveParticles command description:\n"
      "{\n"
      "  \"command\": \"SetParticles\", -- Name of the command, constant\n"
      "  \"particles\": \"p\", -- Particles name from \"Particles\" section\n"
      "  \"geometry\": {}, -- Description of the area, where 'p' will be lost\n"
      "}";
    return help;
  }
};

#endif  // SRC_COMMANDS_BUILDERS_REMOVE_PARTICLES_BUILDER_H
