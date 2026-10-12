#ifndef SRC_DRIFT_KINETIC_SIMULATION_H
#define SRC_DRIFT_KINETIC_SIMULATION_H

#include <petscsnes.h>

#include "src/interfaces/simulation.h"
#include "src/impls/drift_kinetic/diagnostic.h"
#include "src/impls/drift_kinetic/particles.h"

namespace drift_kinetic {

/// @brief Couples guiding-center motion to electromagnetic fields.
class Simulation : public interfaces::Simulation {
public:
  Simulation() = default;
  PetscErrorCode finalize() override;

  std::vector<std::shared_ptr<drift_kinetic::Particles>> particles_;

  Vec M;     // Total magnetization.
  Mat rotM;  // Magnetization curl operator.

protected:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(PetscInt t) override;

  static PetscErrorCode form_iteration(SNES snes, Vec vx, Vec vf, void* ctx);

  PetscErrorCode form_current();
  PetscErrorCode form_function(Vec vf);

  PetscErrorCode from_snes(Vec v, Vec vE, Vec vB);
  PetscErrorCode to_snes(Vec vE, Vec vB, Vec v);

  Vec B_hk;  // Midpoint magnetic field iterate.
  Vec E_hk;  // Midpoint electric field iterate.
  Vec Bn1;   // Predicted next magnetic field.

  Vec Bn_loc;   // Previous magnetic field with ghost cells.
  Vec Bn1_loc;  // Predicted magnetic field with ghost cells.

  Arr Bn_arr;   // Previous magnetic field array.
  Arr Bn1_arr;  // Predicted magnetic field array.

  DM da_EB;  // Combined electric and magnetic field grid.

  Vec sol;                             // Combined field solution.
  SNES snes;                           // Nonlinear field solver.
  PetscInt snes_iteration_number = 0;  // Last field iteration count.

  PetscReal a_MB0 = 0;  // (M * B) at the particle push, for `EnergyConservation`.

  friend class EnergyConservation;
};

}  // namespace drift_kinetic

#endif  // SRC_DRIFT_KINETIC_SIMULATION_H
