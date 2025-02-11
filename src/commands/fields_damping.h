#ifndef SRC_COMMANDS_FIELDS_DAMPING_H
#define SRC_COMMANDS_FIELDS_DAMPING_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/particles.h"
#include "src/utils/geometries.h"

class FieldsDamping : public interfaces::Command {
public:
  using Tester = std::function<bool(const Vector3R&)>;
  using Damping = std::function<void(const Vector3R&, Vector3R&, PetscReal&)>;
  FieldsDamping(DM da, Vec E, Vec B, Vec B0, Tester&& test, Damping&& damp);

  PetscErrorCode execute(timestep_t t) override;

  PetscReal get_damped_energy() const;

private:
  PetscErrorCode damping_implementation(Vec f);

  DM da_;
  Vec E_;
  Vec B_;
  Vec B0_;

  Tester within_geom_;
  Damping damp_;
  PetscReal damped_energy_ = 0.0;
};

class DampForBox {
public:
  void operator()(const Vector3R& g, Vector3R& f, PetscReal& energy);
  BoxGeometry geom;
  PetscReal coefficient;
};

#endif  // SRC_COMMANDS_FIELDS_DAMPING_H
