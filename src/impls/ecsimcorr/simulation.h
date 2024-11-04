#ifndef SRC_ECSIMCORR_SIMULATION_H
#define SRC_ECSIMCORR_SIMULATION_H

#include "src/interfaces/simulation.h"

namespace ecsimcorr {

class Simulation : public interfaces::Simulation {
public:
  Simulation() = default;
  ~Simulation() override;

  // std::vector<Particles> particles_;

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(timestep_t timestep) override;

  Vec E;
  Vec En;
  Vec Ep;
  Vec B;
  Vec B0;
  Vec currI;
  Vec currJ;
  Vec currJe;

  Vec charge_density_old;
  Vec charge_density;

  Mat matL;
  Mat matI;
  Mat rotE;
  Mat rotB;
  Mat rot2EB;
  Mat divE;
};

}  // namespace ecsimcorr

#endif  // SRC_ECSIMCORR_SIMULATION_H
