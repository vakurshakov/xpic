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

  PetscReal w1, w2;

  /// @note Use of stack-allocated `Particles` breaks internal pointers.
  std::vector<std::unique_ptr<Particles>> particles_;

  Vec get_named_vector(std::string_view name) const override;
  const Particles& get_named_particles(std::string_view name) const override;

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
  PetscErrorCode afterprocessing(timestep_t timestep);

  std::unique_ptr<FieldsDamping> damping;
  std::unique_ptr<FieldsEnergy> fields_energy;
  std::unique_ptr<ParticlesEnergy> particles_energy;

  Mat matA;
  Mat matM;

  Mat rotE;
  Mat rotB;

  KSP predict;
  KSP correct;

  PetscLogStage stagenums[6];
};

}  // namespace ecsimcorr

#endif  // SRC_ECSIMCORR_SIMULATION_H
