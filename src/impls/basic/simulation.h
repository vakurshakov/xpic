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

  DM& da();
  Vec& E();
  Vec& B();
  Vec& J();

protected:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(timestep_t timestep) override;

private:
  friend class Diagnostics_builder;

  PetscErrorCode setup_positive_rotor();
  PetscErrorCode setup_negative_rotor();

  PetscInt index(PetscInt k, PetscInt j, PetscInt i, PetscInt l);

  DM da_;
  Vec E_;
  Vec B_;
  Vec J_;
  Mat rot_dt_p;
  Mat rot_dt_m;

  std::vector<Particles> particles_;

  DMBoundaryType bounds[3];
};

}

#endif // SRC_BASIC_SIMULATION_H
