#ifndef SRC_IMPLS_HYBRID_KINETIC_SIMULATION_H
#define SRC_IMPLS_HYBRID_KINETIC_SIMULATION_H

#include "src/impls/drift_kinetic/simulation.h"
#include "src/impls/hybrid_kinetic/particles.h"

namespace hybrid_kinetic {

/// @brief Hybrid model: sorts with `"model": "kinetic"` follow the full
/// Vlasov equation, the rest are drift-kinetic guiding centers. Both are
/// coupled to the fields by the same implicit (E, B) iteration.
class Simulation : public drift_kinetic::Simulation {
public:
  Simulation() = default;
  PetscErrorCode finalize() override;

  std::vector<std::shared_ptr<KineticParticles>> kinetic_particles_;

protected:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(PetscInt t) override;

  static PetscErrorCode form_iteration(SNES snes, Vec vx, Vec vf, void* ctx);

  PetscErrorCode init_kinetic_particles();
  PetscErrorCode form_current();

  PetscErrorCode init_preconditioner();
  static PetscErrorCode form_preconditioned_step(SNES npc, Vec vx);
  PetscErrorCode form_mass_matrix();
  PetscErrorCode check_response();

  Vec x_last;  // Fields of the last push, see `timestep_implementation()`.

  // `"FieldPreconditioner": "mass_matrix"`, see field_preconditioner.md
  bool use_mass_matrix = false;
  PetscBool mass_matrix_every_iteration = PETSC_FALSE;  // `-npc_mass_matrix_every_iteration`.
  PetscBool mass_matrix_check = PETSC_FALSE;            // `-npc_mass_matrix_check`.
  bool mass_matrix_is_current = false;       // Assembled at this time step.
  SNES npc;  // Shell right preconditioner of `snes`.
  KSP ksp;   // Solver of the linearized Maxwell equations.
  Mat matL;  // Mass matrix dt/2 dJ/dE.
  Mat matA = nullptr;  // matM + matL.
  std::vector<PetscReal> coo_v;  // Values of `matL`, blocks by cells.

  friend class EnergyConservation;
};

}  // namespace hybrid_kinetic

#endif  // SRC_IMPLS_HYBRID_KINETIC_SIMULATION_H
