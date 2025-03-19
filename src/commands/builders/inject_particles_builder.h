#ifndef SRC_COMMANDS_BUILDERS_INJECT_PARTICLES_BUILDER_H
#define SRC_COMMANDS_BUILDERS_INJECT_PARTICLES_BUILDER_H

#include "src/commands/builders/particles_builder.h"
#include "src/utils/particles_load.hpp"

class InjectParticlesBuilder : public interfaces::ParticlesBuilder {
public:
  InjectParticlesBuilder(
    interfaces::Simulation& simulation, std::vector<Command_up>& result);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the InjectParticles command description:\n"
      "{\n"
      "  \"command\": \"InjectParticles\", -- Name of the command, constant\n"
      "  \"ionized\": \"Pa\", -- Particles name from \"Particles\" section.\n"
      "  \"ejected\": \"Pb\", -- Particles name from \"Particles\" section.\n"
      "  \"coordinate\": {}, -- Description of _both_ particles coordinate.\n"
      "  \"momentum_i\": {}, -- \"ionized\" particles momentum description.\n"
      "  \"momentum_e\": {}, -- \"ejected\" particles momentum description.\n"
      "}";
    return help;
  }
};

#endif  // SRC_COMMANDS_BUILDERS_INJECT_PARTICLES_BUILDER_H
