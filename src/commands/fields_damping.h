#ifndef SRC_COMMANDS_FIELDS_DAMPING_H
#define SRC_COMMANDS_FIELDS_DAMPING_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/particles.h"
#include "src/utils/geometries.h"

class FieldsDamping : public interfaces::Command {
public:
  /// @todo The same comment from `RemoveParticles` command applies here,
  /// we shouldn't create so many constructors for each geometry type.j
  FieldsDamping(DM da, Vec E, Vec B, Vec B0, //
    const BoxGeometry& geom, PetscReal coefficient);

  FieldsDamping(DM da, Vec E, Vec B, Vec B0, //
    const CircleGeometry& geom, PetscReal coefficient);

  PetscErrorCode execute(timestep_t t) override;

  PetscReal get_damped_energy() const;

private:
  PetscErrorCode damping_implementation(Vec f);

  DM da_;
  Vec E_;
  Vec B_;
  Vec B0_;

  using Damping =
    std::function<void(PetscInt, PetscInt, PetscInt, Vector3R***, PetscReal&)>;

  Damping damp_;
  PetscReal damped_energy_ = 0.0;

  class DampForBox;
  class DampForCircle;
};

#endif  // SRC_COMMANDS_FIELDS_DAMPING_H
