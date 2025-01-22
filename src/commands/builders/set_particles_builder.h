#ifndef SRC_COMMANDS_BUILDERS_SET_PARTICLES_BUILDER_H
#define SRC_COMMANDS_BUILDERS_SET_PARTICLES_BUILDER_H

#include "src/commands/builders/command_builder.h"
#include "src/utils/particles_load.hpp"

class SetParticlesBuilder : public CommandBuilder {
public:
  SetParticlesBuilder(
    const interfaces::Simulation& simulation, std::list<Command_up>& result);

  PetscErrorCode build(const Configuration::json_t& info) override;

protected:
  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the SetParticles diagnostic description:\n"
      "{\n"
      "  \"command\": \"SetParticles\", -- Name of the comand, constant\n"
      "  \"particles\": \"p\", -- Particles name from \"Particles\" section.\n"
      "  \"set_point_of_birth\": {}, -- Description of particles\n"
      "                                 coordinate generator.\n"
      "  \"load_momentum\": {}, -- Description of particles\n"
      "                            momentum generator.\n"
      "}";
    return help;
  }

  void load_coordinate(const Configuration::json_t& info,
    const interfaces::Particles& particles, CoordinateGenerator& gen,
    PetscInt& per_step_particles_number);

  void load_momentum(const Configuration::json_t& info,
    const interfaces::Particles& particles, MomentumGenerator& gen);
};

#endif  // SRC_COMMANDS_BUILDERS_SET_PARTICLES_BUILDER_H
