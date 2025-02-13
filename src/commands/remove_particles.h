#ifndef SRC_COMMANDS_REMOVE_PARTICLES_H
#define SRC_COMMANDS_REMOVE_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/particles.h"
#include "src/utils/geometries.h"


class RemoveParticles : public interfaces::Command {
public:
  /// @todo `RemoveParticles::Tester` should be made public,
  /// we should create structs like it is in `SetParticles`.
  RemoveParticles(interfaces::Particles& particles, const BoxGeometry& geom);
  RemoveParticles(interfaces::Particles& particles, const CircleGeometry& geom);

  PetscErrorCode execute(timestep_t t) override;

  std::string get_particles_name() const;
  PetscReal get_removed_energy() const;

private:
  interfaces::Particles& particles_;

  using Tester = std::function<bool(const Point& /* point */)>;
  Tester should_remove_;
  PetscReal removed_energy_ = 0.0;

  class RemoveFromBox;
  class RemoveFromCircle;
};

#endif  // SRC_COMMANDS_REMOVE_PARTICLES_H
