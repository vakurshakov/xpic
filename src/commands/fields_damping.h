#ifndef SRC_COMMANDS_FIELDS_DAMPING_H
#define SRC_COMMANDS_FIELDS_DAMPING_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/particles.h"
#include "src/utils/geometries.h"

class FieldsDamping : public interfaces::Command {
public:
  FieldsDamping(DM da, const std::vector<Vec>& storages,
    const CircleGeometry& geom, PetscReal coefficient);

  PetscErrorCode execute(timestep_t t) override;

  PetscReal get_damped_energy() const;

private:
  DM da_;
  std::vector<Vec> storages_;

  using Damping =
    std::function<void(PetscInt, PetscInt, PetscInt, Vector3R***, PetscReal&)>;

  Damping damp_;

  PetscReal damped_energy_;

  class DampForCircle;
};

#endif  // SRC_COMMANDS_FIELDS_DAMPING_H
