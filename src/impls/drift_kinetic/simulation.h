#ifndef SRC_DRIFT_KINETIC_SIMULATION_H
#define SRC_DRIFT_KINETIC_SIMULATION_H

#include <petscsnes.h>

#include "src/interfaces/simulation.h"
#include "src/diagnostics/utils/table_diagnostic.h"
#include "src/impls/drift_kinetic/particles.h"
#include "src/utils/sync_clock.h"

namespace drift_kinetic {

class EnergyConservation;

class Simulation : public interfaces::Simulation {
public:
  Simulation() = default;
  PetscErrorCode finalize() override;

  std::vector<std::unique_ptr<drift_kinetic::Particles>> particles_;

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

  Vec M;

  Vec dBdx;
  Vec dBdx_loc;
  Arr dBdx_arr;

  Vec dBdy;
  Vec dBdy_loc;
  Arr dBdy_arr;

  Vec dBdz;
  Vec dBdz_loc;
  Arr dBdz_arr;

  Vec B_hk;
  Vec E_hk;

  DM da_EB;
  Vec sol;
  SNES snes;

  friend class EnergyConservation;
  std::unique_ptr<EnergyConservation> energy_cons;
};

class EnergyConservation : public TableDiagnostic {
public:
  EnergyConservation(const Simulation& simulation);
  PetscErrorCode diagnose(PetscInt t) override;
  PetscErrorCode add_columns(PetscInt t) override;

  const Simulation& simulation;
  PetscReal w_E = 0, w_E0 = 0;
  PetscReal w_B = 0, w_B0 = 0;
  PetscReal dF = 0;
  PetscReal a_EJ = 0;
  PetscReal a_MB = 0, a_MB0 = 0;
};

}  // namespace drift_kinetic

#endif  // SRC_DRIFT_KINETIC_SIMULATION_H
