#ifndef SRC_COMMANDS_BUILDERS_INJECT_PARTICLES_BUILDER_H
#define SRC_COMMANDS_BUILDERS_INJECT_PARTICLES_BUILDER_H

#include "src/commands/builders/set_particles_builder.h"
#include "src/utils/particles_load.hpp"

class InjectParticlesBuilder : public SetParticlesBuilder {
public:
  InjectParticlesBuilder(
    const interfaces::Simulation& simulation, std::list<Command_up>& result);

  PetscErrorCode build(const Configuration::json_t& info) override;

private:
  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the InjectParticles diagnostic description:\n"
      "{\n"
      "  \"command\": \"InjectParticles\", -- Name of the comand, constant\n"
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
