#ifndef SRC_COMMANDS_BUILDERS_PARTICLES_BUILDER_H
#define SRC_COMMANDS_BUILDERS_PARTICLES_BUILDER_H

#include "src/commands/builders/command_builder.h"
#include "src/utils/particles_load.hpp"

namespace interfaces {

class ParticlesBuilder : public CommandBuilder {
public:
  ParticlesBuilder(
    interfaces::Simulation& simulation, std::vector<Command_up>& result);

protected:
  void load_coordinate(const Configuration::json_t& info,
    const interfaces::Particles& particles, CoordinateGenerator& gen,
    PetscInt& number_of_particles);

  void load_momentum(const Configuration::json_t& info,
    const interfaces::Particles& particles, MomentumGenerator& gen);
};

}  // namespace interfaces

#endif  // SRC_COMMANDS_BUILDERS_PARTICLES_BUILDER_H
