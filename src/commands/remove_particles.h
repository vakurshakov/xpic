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

  PetscReal get_removed_energy() const;

private:
  interfaces::Particles& particles_;

  using Tester = std::function<bool(const Point& /* point */)>;
  Tester should_remove_;
  PetscReal removed_energy_;

  class RemoveFromCircle;
};

#endif  // SRC_COMMANDS_REMOVE_PARTICLES_H
