#ifndef SRC_COMMANDS_REMOVE_PARTICLES_H
#define SRC_COMMANDS_REMOVE_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/particles.h"
#include "src/utils/geometries.h"


class RemoveParticles : public interfaces::Command {
public:
  using Tester = std::function<bool(const Point& /* point */)>;
  RemoveParticles(interfaces::Particles& particles, Tester&& test);

  PetscErrorCode execute(timestep_t t) override;

  std::string get_particles_name() const;
  PetscReal get_removed_energy() const;

private:
  interfaces::Particles& particles_;

  Tester should_remove_;
  PetscReal removed_energy_ = 0.0;
};


struct RemoveFromBox {
  bool operator()(const Point& point);
  BoxGeometry geom_;
};

struct RemoveFromCircle {
  bool operator()(const Point& point);
  CircleGeometry geom_;
};

struct RemoveFromCylinder {
  bool operator()(const Point& point);
  CylinderGeometry geom_;
};


#endif  // SRC_COMMANDS_REMOVE_PARTICLES_H
