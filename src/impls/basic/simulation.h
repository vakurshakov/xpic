#ifndef SRC_BASIC_SIMULATION_H
#define SRC_BASIC_SIMULATION_H

#include "src/interfaces/simulation.h"

#include <petscdmda.h>
#include <petscvec.h>
#include <petscmat.h>

#include "src/impls/basic/particles.h"

namespace basic {

class Simulation : public interfaces::Simulation {
public:
  Simulation() = default;
  ~Simulation() override;

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(timestep_t timestep) override;

  PetscErrorCode setup_positive_rotor();
  PetscErrorCode setup_negative_rotor();

  PetscInt index(PetscInt k, PetscInt j, PetscInt i, PetscInt l);

  friend class Particles;

  // evil-evil bad-bad-bad
  friend class Diagnostic_builder;
  friend class Fields_energy_builder;
  friend class Field_view_builder;

  DM da_;
  DMBoundaryType bounds[3];

  Vec E_;
  Vec B_;
  Vec J_;
  Mat rot_dt_p;
  Mat rot_dt_m;

  std::vector<Particles> particles_;
};

}

#endif // SRC_BASIC_SIMULATION_H
