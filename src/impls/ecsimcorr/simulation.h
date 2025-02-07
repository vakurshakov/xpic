#ifndef SRC_ECSIMCORR_SIMULATION_H
#define SRC_ECSIMCORR_SIMULATION_H

#include <petscksp.h>

#include "src/interfaces/simulation.h"
#include "src/impls/ecsimcorr/particles.h"

namespace ecsimcorr {

class Simulation : public interfaces::Simulation {
public:
  DEFAULT_MOVABLE(Simulation);

  Simulation() = default;
  ~Simulation() override;

  Vec E;
  Vec Ep;
  Vec Ec;
  Vec B;
  Vec B0;
  Vec currI;
  Vec currJe;

  /**
   * @details Filling up the Lapenta's matrix will be one of the most time
   * consuming part of the simulation process, so a proper way to do so should
   * be chosen. In attempts to speed up this process, we use the following steps:
   *
   * 1) `MatSetPreallocationCOO()`/`MatSetValuesCOO()` technique is utilized;
   * 2) Control of matrix indices assembly is added (to avoid reallocation);
   * 3) Parallel-traversable buffers are used to fill indices and values.
   */
  Mat matL;

  /// @note Use of stack-allocated `Particles` breaks internal pointers.
  std::vector<std::unique_ptr<Particles>> particles_;

  Vec get_named_vector(std::string_view name) override;
  Particles& get_named_particles(std::string_view name) override;

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(timestep_t t) override;

  PetscErrorCode init_vectors();
  PetscErrorCode init_matrices();
  PetscErrorCode init_ksp_solvers();
  PetscErrorCode init_log_stages();

  PetscErrorCode clear_sources();
  PetscErrorCode predict_fields();
  PetscErrorCode correct_fields();
  PetscErrorCode final_update();

  PetscErrorCode advance_fields(KSP ksp, Vec curr, Vec out);

  PetscErrorCode update_cells();
  PetscErrorCode fill_ecsim_current();

  PetscErrorCode mat_set_preallocation_coo(
    PetscInt size, MatStencil* coo_i, MatStencil* coo_j);

  Mat matA;
  Mat matM;

  Mat rotE;
  Mat rotB;

  KSP predict;
  KSP correct;

  /// @note We start with _unassembled_ state to force first indexes assembly
  bool matL_indices_assembled = false;

  PetscLogStage stagenums[6];

  friend class EnergyConservation;
};

}  // namespace ecsimcorr

#endif  // SRC_ECSIMCORR_SIMULATION_H
