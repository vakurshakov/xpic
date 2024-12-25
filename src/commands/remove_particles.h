#ifndef SRC_COMMANDS_REMOVE_PARTICLES_H
#define SRC_COMMANDS_REMOVE_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/particles.h"

struct CircleGeometry {
  CircleGeometry(const Vector3R& center, PetscReal radius);

  Vector3R center;
  PetscReal radius;
};

class RemoveParticles : public interfaces::Command {
public:
  RemoveParticles(interfaces::Particles& particles, const CircleGeometry& geom);

  PetscErrorCode execute(timestep_t t) override;

private:
  interfaces::Particles& particles_;

  using Remove = std::function<PetscErrorCode(const Point& /* point */)>;
  Remove remove_;

  class RemoveFromCircle {
  public:
    RemoveFromCircle(const CircleGeometry& geom);
    bool operator()(const Point& point);
    CircleGeometry geom_;
  };
};

#endif  // SRC_COMMANDS_REMOVE_PARTICLES_H
