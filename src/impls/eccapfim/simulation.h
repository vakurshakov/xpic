#ifndef SRC_ECCAPFIM_SIMULATION_H
#define SRC_ECCAPFIM_SIMULATION_H

#include <petscsnes.h>

#include "src/interfaces/simulation.h"
#include "src/impls/eccapfim/particles.h"
#include "src/utils/sync_clock.h"

namespace eccapfim {

#define SNES_ITERATE_B 0

class Simulation : public interfaces::Simulation {
public:
  Simulation() = default;
  PetscErrorCode finalize() override;

  std::vector<std::shared_ptr<Particles>> particles_;

protected:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(PetscInt t) override;

  PetscErrorCode init_vectors();
  PetscErrorCode init_matrices();
  PetscErrorCode init_snes_solver();
  PetscErrorCode init_log_stages();

  // Iterative solution procedures
  PetscErrorCode init_iteration();
  PetscErrorCode calc_iteration();
  PetscErrorCode after_iteration();

  /**
   * @brief Evaluates nonlinear function F(x^k), namely the system of
   * Maxwell's equations on E^{n+1/2,k}, B^{n+1/2,k}. In fact, this is
   * a main "computational step".
   *
   * @param[in]  snes the SNES context.
   * @param[in]  vx   input vector of (E^{n+1/2}, B^{n+1/2}) on k-th iteration.
   * @param[in]  ctx  user context, we pass `this` i.e. `Simulation*`.
   * @param[out] vf   function to be evaluated, system of Maxwell's equations.
   */
  static PetscErrorCode form_iteration(SNES snes, Vec vx, Vec vf, void* ctx);

  // The main simulation steps
  PetscErrorCode clear_sources();
  PetscErrorCode form_current();
  PetscErrorCode form_function(Vec vf);

#if SNES_ITERATE_B
  DM da_EB;
  Vec B_hk;
  PetscErrorCode from_snes(Vec v, Vec vE, Vec vB);
  PetscErrorCode to_snes(Vec vE, Vec vB, Vec v);
#else
  Mat matM;
#endif

  Vec J;
  Vec E_hk;
  Vec E_loc;
  Vec B_loc;

  Vector3R*** E_arr;
  Vector3R*** B_arr;

  Mat rotE;
  Mat rotB;

  Vec sol;
  SNES snes;
  std::vector<PetscReal> conv_hist;

  PetscClassId classid;
  PetscLogEvent events[5];
  PetscLogStage stagenums[4];

  SyncClock clock;

  friend class ConvergenceHistory;
};

}  // namespace eccapfim

#endif  // SRC_ECCAPFIM_SIMULATION_H
