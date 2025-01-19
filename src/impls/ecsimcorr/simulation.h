#ifndef SRC_ECSIMCORR_SIMULATION_H
#define SRC_ECSIMCORR_SIMULATION_H

#include <petscksp.h>

#include "src/interfaces/simulation.h"
#include "src/commands/fields_damping.h"
#include "src/diagnostics/fields_energy.h"
#include "src/diagnostics/particles_energy.h"
#include "src/impls/ecsimcorr/particles.h"

namespace ecsimcorr {

class Simulation : public interfaces::Simulation {
public:
  DEFAULT_MOVABLE(Simulation);

  Simulation() = default;
  ~Simulation() override;

  Vec E;
  Vec En;
  Vec B;
  Vec B0;
  Vec currI;
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

  /// @note Use of stack-allocated `Particles` breaks internal pointers.
  std::vector<std::unique_ptr<Particles>> particles_;

  Vec get_named_vector(std::string_view name) override;
  Particles& get_named_particles(std::string_view name) override;

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(timestep_t timestep) override;

  PetscErrorCode init_vectors();
  PetscErrorCode init_matrices();
  PetscErrorCode init_ksp_solvers();
  PetscErrorCode init_log_stages();

  PetscErrorCode clear_sources();
  PetscErrorCode predict_fields();
  PetscErrorCode correct_fields();
  PetscErrorCode advance_fields(KSP ksp, Vec rhs);
  PetscErrorCode final_update();

  std::unique_ptr<FieldsDamping> damping;

  Mat matA;
  Mat matM;

  Mat rotE;
  Mat rotB;

  KSP predict;
  KSP correct;

  PetscLogStage stagenums[6];

  friend class EnergyDiagnostic;
};

}  // namespace ecsimcorr

#endif  // SRC_ECSIMCORR_SIMULATION_H
