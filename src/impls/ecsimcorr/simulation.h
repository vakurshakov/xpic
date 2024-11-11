#ifndef SRC_ECSIMCORR_SIMULATION_H
#define SRC_ECSIMCORR_SIMULATION_H

#include <petscksp.h>

#include "src/interfaces/simulation.h"
#include "src/impls/ecsimcorr/particles.h"

namespace ecsimcorr {

class Simulation : public interfaces::Simulation {
public:
  Simulation() = default;
  ~Simulation() override;

  Vec E;
  Vec En;
  Vec Ep;
  Vec B;
  Vec B0;
  Vec currI;
  Vec currJ;
  Vec currJe;

  /**
   * @details Filling up the (L)apenta's matrix will be the most time consuming
   * part of the simulation process, so a proper path to do so should be chosen.
   * In attempts to achieve it, we use the following steps:
   *
   * (1) We treat `matL` as PETSc `baij` matrix and use `MatSetValuesBlocked()`,
   *     since the blocksize of 3 (dof) was already set in `DMCreateMatrix()`.
   *
   * (2) We _should_ store nonzeros structure of the matrix and use the sequence
   *     when needed: `DMSetMatrixPreallocateSkip()`, `MatSetPreallocation()`.
   */
  Mat matL;

  std::vector<Particles> particles_;

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(timestep_t timestep) override;

  PetscErrorCode clear_sources();
  PetscErrorCode predict_E();

  // Vec charge_density_old;
  // Vec charge_density;

  Mat rotE;
  Mat rotB;
  Mat rot2BE;
  Mat divE;

  KSP ksp;
};

}  // namespace ecsimcorr

#endif  // SRC_ECSIMCORR_SIMULATION_H
