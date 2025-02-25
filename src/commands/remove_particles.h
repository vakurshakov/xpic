#ifndef SRC_COMMANDS_REMOVE_PARTICLES_H
#define SRC_COMMANDS_REMOVE_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/particles.h"
#include "src/utils/geometries.h"


class RemoveParticles : public interfaces::Command {
public:
  using Tester = std::function<bool(const Vector3R& g)>;
  RemoveParticles(interfaces::Particles& particles, Tester&& test);

  PetscErrorCode execute(PetscInt t) override;

  std::string get_particles_name() const;
  PetscReal get_removed_energy() const;

private:
  PetscErrorCode log_statistics();

  interfaces::Particles& particles_;

  Tester within_geom_;

  PetscReal removed_energy_ = 0.0;
  PetscInt removed_particles_ = 0;
};

#endif  // SRC_COMMANDS_REMOVE_PARTICLES_H
