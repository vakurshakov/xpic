#ifndef SRC_COMMANDS_REMOVE_PARTICLES_H
#define SRC_COMMANDS_REMOVE_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/particles.h"
#include "src/utils/geometries.h"


class RemoveParticles : public interfaces::Command {
public:
  RemoveParticles(interfaces::Particles& particles, const CircleGeometry& geom);

  PetscErrorCode execute(timestep_t t) override;

private:
  interfaces::Particles& particles_;

  using Remove = std::function<PetscErrorCode(const Point& /* point */)>;
  Remove remove_;

  class RemoveFromCircle;
};

#endif  // SRC_COMMANDS_REMOVE_PARTICLES_H
