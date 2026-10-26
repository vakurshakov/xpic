#ifndef SRC_COMMANDS_BUILDERS_SET_PARTICLES_BUILDER_H
#define SRC_COMMANDS_BUILDERS_SET_PARTICLES_BUILDER_H

#include "src/commands/builders/particles_builder.h"
#include "src/commands/set_particles.h"

class SetParticlesBuilder : public ParticlesBuilder {
public:
  SetParticlesBuilder(
    interfaces::Simulation& simulation, std::vector<Command_up>& result);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the SetParticles command description:\n"
      "{\n"
      "  \"command\": \"SetParticles\", -- Name of the command, constant\n"
      "  \"particles\": \"p\", -- Particles name from \"Particles\" section.\n"
      "  \"coordinate\": {}, -- Particles coordinate generator description.\n"
      "  \"momentum\": {}, -- Particles momentum generator description.\n"
      "  \"paired_with\": \"q\", -- Optional second particles name, it shares\n"
      "                          the coordinate stream with \"particles\".\n"
      "  \"momentum_paired\": {}, -- Momentum of \"paired_with\" particles.\n"
      "}";
    return help;
  }

protected:
  struct SortInfo {
    std::string name;
    const Configuration::json_t& momentum_info;
  };

  std::vector<SetParticles::Sort> load_sorts(      //
    const Configuration::json_t& coordinate_info,  //
    const std::vector<SortInfo>& sort_infos,       //
    CoordinateGenerator& generate_coordinate,      //
    PetscInt& number_of_particles);
};

#endif  // SRC_COMMANDS_BUILDERS_SET_PARTICLES_BUILDER_H
