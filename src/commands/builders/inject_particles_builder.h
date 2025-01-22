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
      "  \"set_point_of_birth\": {}, -- Description of _both_ particles\n"
      "                                 coordinate generator.\n"
      "  \"load_momentum_i\": {}, -- Description of \"ionized\" particles\n"
      "                              momentum generator.\n"
      "  \"load_momentum_e\": {}, -- Description of \"ejected\" particles\n"
      "                              momentum generator.\n"
      "}";
    return help;
  }
};

#endif  // SRC_COMMANDS_BUILDERS_INJECT_PARTICLES_BUILDER_H
