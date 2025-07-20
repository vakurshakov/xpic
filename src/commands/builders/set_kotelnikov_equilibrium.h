#ifndef SRC_COMMANDS_BUILDERS_SET_K_EQ_H
#define SRC_COMMANDS_BUILDERS_SET_K_EQ_H

#include "src/commands/builders/particles_builder.h"
#include "src/utils/particles_load.h"

namespace kotelnikov_equilibrium {

class SetPresets : public ::CommandBuilder {
public:
  SetPresets(interfaces::Simulation& simulation, std::vector<Command_up>&);
  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\n"
      "To use the \"SetKotelnikovEquilibrium\" preset it is needed to run the\n"
      "python script generating the distribution caches first. It's placed at\n"
      "[...]/kotelnikov_equilibrium/cyl_without_phi/py-scripts/calc_maxw.py.\n"
      "Once caches are generated, \"parameters\" option matching the result \n"
      "directory have to be specified in config to initialize the xpic.\n";
    return help;
  }
};

class SetStepPresets : public ParticlesBuilder {
public:
  SetStepPresets(interfaces::Simulation& simulation, std::vector<Command_up>&);
  PetscErrorCode build(const Configuration::json_t& info) override;

  std::string_view usage_message() const override
  {
    std::string_view help =
      "\n"
      "\"SetKotelnikovEquilibrium\" step preset description:\n"
      "\"buffer\" -- width of buffer to leave along cylinder radius\n"
      "\"damping_coefficient\" -- the coefficient of \"FieldsDamping\"\n";
    return help;
  }
};

}  // namespace kotelnikov_equilibrium

#endif  // SRC_COMMANDS_BUILDERS_SET_K_EQ_H
