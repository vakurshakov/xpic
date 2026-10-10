#ifndef SRC_ECCAPFIM_SIMULATION_H
#define SRC_ECCAPFIM_SIMULATION_H

#include <petscsnes.h>

#include "src/interfaces/simulation.h"
#include "src/diagnostics/table_diagnostic.h"
#include "src/impls/eccapfim/particles.h"
#include "src/utils/sync_clock.h"

/// @todo Try to produce more information about crashes, see `PetscUnlikely`

namespace eccapfim {

#define SNES_ITERATE_B 0
#define SNES_PRECONDITIONING 1

static constexpr PetscReal atol = 1e-7;
static constexpr PetscReal rtol = 1e-7;
static constexpr PetscReal stol = 1e-7;
static constexpr PetscReal divtol = PETSC_DETERMINE;
static constexpr PetscInt maxit = 1000;
static constexpr PetscInt maxf = PETSC_UNLIMITED;

static constexpr PetscInt ew_version = 3;
static constexpr PetscReal ew_rtol_0 = 0.8;
static constexpr PetscReal ew_gamma = 0.9;
static constexpr PetscReal ew_alpha = 1.5;

class Simulation : public interfaces::Simulation {
public:
  Simulation() = default;
  PetscErrorCode finalize() override;

  std::vector<std::shared_ptr<Particles>> particles_;

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(PetscInt t) override;

  PetscErrorCode init_snes();
  PetscErrorCode init_log();

  // Iterative solution procedures
  PetscErrorCode init_iteration();
  PetscErrorCode calc_iteration();
  PetscErrorCode after_iteration();

  static PetscErrorCode form_prediction(SNES npc, Vec x);
  PetscErrorCode impl_form_prediction(Vec vx);

  /**
   * @brief Evaluates nonlinear function F(x^k), namely the system of
   * Maxwell's equations on E^{n+1/2,k}, B^{n+1/2,k}. This is a main
   * computational step.
   *
   * @param[in]  snes the SNES context.
   * @param[in]  x    input vector of (E^{n+1/2}, B^{n+1/2}) on k-th iteration.
   * @param[in]  ctx  user context, we pass `this` i.e. `Simulation*`.
   * @param[out] f    function to be evaluated, system of Maxwell's equations.
   */
  static PetscErrorCode form_iteration(SNES snes, Vec x, Vec f, void* ctx);
  PetscErrorCode impl_form_iteration(Vec x, Vec f);

  static PetscErrorCode mat_mult(Mat mat, Vec x, Vec y);
  PetscErrorCode impl_mat_mult(Vec x, Vec y);

  PetscErrorCode clear_sources();
  PetscErrorCode form_current();
  PetscErrorCode form_function(Vec f);

  PetscErrorCode comp_snes(Vec vE, Vec vB, Vec v);
  PetscErrorCode decomp_snes(Vec vE, Vec vB, Vec v);

  // `da` can be extended, if we iterate magnetic and electric field together
  SNES snes;
  DM sda;
  Vec sol;
  Vec E_hk;
  Vec B_hk;

  // Simplified matrix storage, used for preconditioning if needed
  SNES npc;
  KSP ksp;
  DM nda;
  Vec nsol;

  Vec I;
  Vec I_loc;
  PetscReal**** I_arr;

  Vec nLv;
  Vec nLv_loc;
  PetscReal**** nLv_arr;

  Mat nLm;

  std::vector<PetscReal> conv_hist;

  PetscClassId classid;
  PetscLogEvent events[7];
  PetscLogStage stagenums[4];

  SyncClock clock;

  friend class ConvergenceHistory;
};


class ConvergenceHistory : public TableDiagnostic {
public:
  ConvergenceHistory(const Simulation& simulation);
  PetscErrorCode add_columns(PetscInt t) override;
  const Simulation& simulation;
};

}  // namespace eccapfim

#endif  // SRC_ECCAPFIM_SIMULATION_H
