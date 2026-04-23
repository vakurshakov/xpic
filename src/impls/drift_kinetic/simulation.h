#ifndef SRC_DRIFT_KINETIC_SIMULATION_H
#define SRC_DRIFT_KINETIC_SIMULATION_H

#include <petscsnes.h>

#include "src/interfaces/simulation.h"
#include "src/diagnostics/table_diagnostic.h"
#include "src/impls/drift_kinetic/particles.h"
#include "src/utils/sync_clock.h"
#include "src/impls/drift_kinetic/diagnostic.h"

namespace drift_kinetic {

class Simulation : public interfaces::Simulation {
public:
  Simulation() = default;
  PetscErrorCode finalize() override;

  std::vector<std::shared_ptr<drift_kinetic::Particles>> particles_;

  Vec M;

protected:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(PetscInt t) override;

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

  PetscErrorCode form_current();
  PetscErrorCode form_function(Vec vf);

  PetscErrorCode from_snes(Vec v, Vec vE, Vec vB);
  PetscErrorCode to_snes(Vec vE, Vec vB, Vec v);


  Vec B_hk;
  Vec E_hk;

  Vec B0_loc;
  Arr B0_arr;

  DM da_EB;

  Vec sol;
  SNES snes;
  PetscInt last_field_itnum = 0;

  Mat rotM;

  friend class EnergyConservation;
  friend class Particles;
  std::unique_ptr<EnergyConservation> energy_cons;
  std::unique_ptr<PointByFieldTrace> trace;
};

}  // namespace drift_kinetic

#endif  // SRC_DRIFT_KINETIC_SIMULATION_H
