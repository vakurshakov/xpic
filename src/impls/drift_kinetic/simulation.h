#ifndef SRC_DRIFT_KINETIC_SIMULATION_H
#define SRC_DRIFT_KINETIC_SIMULATION_H

#include <petscsnes.h>

#include "src/interfaces/simulation.h"
#include "src/impls/drift_kinetic/particles.h"
#include "src/impls/drift_kinetic/diagnostic.h"

/// @file simulation.h
/// @brief Implicit drift-kinetic simulation.

namespace drift_kinetic {

/// @brief Couples guiding-center motion to electromagnetic fields.
class Simulation : public interfaces::Simulation {
public:
  /// @brief Creates an uninitialized simulation.
  Simulation() = default;
  /// @brief Finishes diagnostics and releases simulation resources.
  PetscErrorCode finalize() override;

  /// @brief Drift-kinetic particle sorts shared with the base simulation.
  std::vector<std::shared_ptr<drift_kinetic::Particles>> particles_;

  Vec M;     ///< Total magnetization.
  Mat rotM;  ///< Magnetization curl operator.

protected:
  /// @brief Initializes field iteration, particle sorts and diagnostics.
  PetscErrorCode initialize_implementation() override;
  /// @brief Solves one timestep and updates particle ownership.
  PetscErrorCode timestep_implementation(PetscInt t) override;

  /// @brief SNES callback collecting sources and evaluating field residuals.
  static PetscErrorCode form_iteration(SNES snes, Vec vx, Vec vf, void* ctx);

  /// @brief Advances particles and collects current and magnetization.
  PetscErrorCode form_current();
  /// @brief Evaluates Maxwell residuals for the current field iterate.
  PetscErrorCode form_function(Vec vf);

  /// @brief Unpacks the SNES vector into electric and magnetic fields.
  PetscErrorCode from_snes(Vec v, Vec vE, Vec vB);
  /// @brief Packs electric and magnetic fields into the SNES vector.
  PetscErrorCode to_snes(Vec vE, Vec vB, Vec v);

  Vec B_hk;  ///< Midpoint magnetic field iterate.
  Vec E_hk;  ///< Midpoint electric field iterate.
  Vec Bn1;   ///< Predicted next magnetic field.

  Vec Bn_loc;   ///< Previous magnetic field with ghost cells.
  Vec Bn1_loc;  ///< Predicted magnetic field with ghost cells.

  Arr Bn_arr;   ///< Previous magnetic field array.
  Arr Bn1_arr;  ///< Predicted magnetic field array.

  DM da_EB;  ///< Combined electric and magnetic field grid.

  Vec sol;   ///< Combined field solution.
  SNES snes;  ///< Nonlinear field solver.
  PetscInt last_field_itnum = 0;  ///< Last field iteration count.

  friend class EnergyConservation;
  std::unique_ptr<EnergyConservation> energy_cons;  ///< Energy diagnostic.
  std::unique_ptr<PointByFieldTrace> trace;         ///< Particle trace.
};

}  // namespace drift_kinetic

#endif  // SRC_DRIFT_KINETIC_SIMULATION_H
