#ifndef SRC_ECSIMCORR_SIMULATION_H
#define SRC_ECSIMCORR_SIMULATION_H

#include <petscksp.h>

#include "src/interfaces/simulation.h"
#include "src/impls/ecsimcorr/particles.h"
#include "src/utils/sync_clock.h"

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

  NamedValues<Vec> get_backup_fields() override;
  NamedValues<interfaces::Particles*> get_backup_particles() override;

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(PetscInt t) override;

  PetscErrorCode init_vectors();
  PetscErrorCode init_matrices();
  PetscErrorCode init_ksp_solvers();
  PetscErrorCode init_particles();
  PetscErrorCode init_log_stages();

  // The main simulation steps
  PetscErrorCode clear_sources();
  PetscErrorCode first_push();
  PetscErrorCode fill_ecsim_current();
  PetscErrorCode predict_fields();
  PetscErrorCode second_push();
  PetscErrorCode correct_fields();
  PetscErrorCode final_update();

  PetscErrorCode update_cells_with_assembly();
  PetscErrorCode advance_fields(KSP ksp, Vec curr, Vec out);

  void get_array_offset(PetscInt begin_g, PetscInt end_g, PetscInt& off);

  PetscErrorCode fill_matrix_indices(MatStencil* coo_i, MatStencil* coo_j);
  PetscErrorCode fill_ecsim_current(PetscReal* coo_v);

  PetscErrorCode mat_set_preallocation_coo(
    PetscInt size, MatStencil* coo_i, MatStencil* coo_j);

  PetscErrorCode log_timings();

  Vec local_E;
  Vec local_B;

  Mat rotE;
  Mat rotB;
  Mat matM;

  KSP predict;
  KSP correct;

  /**
   * @brief Radius of the cloud, where indices should be assembled.
   * @details The available values are:
   * 1) radius = 0 -- All cells where _were_ particles are assembled.
   * 2) radius > 0 -- Adds a `radius` cells along each direction to
   *                  where the particles _were_ placed.
   * 3) radius < 0 -- Drop the previous `assembly_map` in this case.
   */
  static constexpr PetscInt assembly_radius = -1;
  static constexpr PetscInt assembly_width = 2 * assembly_radius + 1;

  /// @brief Cells, where indices have been assembled.
  /// @note This map and `matL` size can only _grow_ in time.
  std::vector<bool> assembly_map;

  /// @brief Whether the new cells have been added into `indices_map`.
  bool indices_assembled = false;

  PetscClassId classid;
  PetscLogEvent events[1];
  PetscLogStage stagenums[7];

  SyncClock clock;

  friend class EnergyConservation;
};

}  // namespace ecsimcorr

#endif  // SRC_ECSIMCORR_SIMULATION_H
