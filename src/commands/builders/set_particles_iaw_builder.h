#ifndef SRC_COMMANDS_BUILDERS_SET_PARTICLES_IAW_BUILDER_H
#define SRC_COMMANDS_BUILDERS_SET_PARTICLES_IAW_BUILDER_H

#include "src/commands/builders/command_builder.h"

class SetParticlesIonAcousticWaveBuilder : public CommandBuilder {
public:
  SetParticlesIonAcousticWaveBuilder(
    interfaces::Simulation& simulation, std::vector<Command_up>& result);

  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\nStructure of the SetParticlesIonAcousticWave command description:\n"
      "{\n"
      "  \"command\": \"SetParticlesIonAcousticWave\", -- Name of the command\n"
      "  \"particles\": \"p\", -- Particles name from \"Particles\" section.\n"
      "  \"wave_number\": [], -- Integer mode number, one non-zero component.\n"
      "  \"density_amplitude\": [], -- Relative density perturbation.\n"
      "  \"density_phase\": [], -- Optional, density perturbation phase.\n"
      "  \"force_electric_amplitude\": 0, -- Electric field amplitude.\n"
      "  \"field_phase\": [], -- Optional, electric field phase.\n"
      "  \"omega_real\": 0, -- Real part of the mode frequency.\n"
      "  \"gamma\": 0, -- Mode damping rate, positive.\n"
      "}";
    return help;
  }
};

#endif  // SRC_COMMANDS_BUILDERS_SET_PARTICLES_IAW_BUILDER_H
