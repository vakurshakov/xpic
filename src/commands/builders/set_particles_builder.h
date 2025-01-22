#ifndef SRC_COMMANDS_BUILDERS_SET_PARTICLES_BUILDER_H
#define SRC_COMMANDS_BUILDERS_SET_PARTICLES_BUILDER_H

#include "src/commands/builders/particles_builder.h"
#include "src/utils/particles_load.hpp"

class SetParticlesBuilder : public interfaces::ParticlesBuilder {
public:
  SetParticlesBuilder(
    const interfaces::Simulation& simulation, std::list<Command_up>& result);

  PetscErrorCode build(const Configuration::json_t& info) override;

protected:
  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the SetParticles command description:\n"
      "{\n"
      "  \"command\": \"SetParticles\", -- Name of the command, constant\n"
      "  \"particles\": \"p\", -- Particles name from \"Particles\" section.\n"
      "  \"coordinate\": {}, -- Particles coordinate generator description.\n"
      "  \"momentum\": {}, -- Particles momentum generator description.\n"
      "}";
    return help;
  }
};

#endif  // SRC_COMMANDS_BUILDERS_SET_PARTICLES_BUILDER_H
