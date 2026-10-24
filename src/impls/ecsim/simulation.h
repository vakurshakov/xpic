#ifndef SRC_ECSIM_SIMULATION_H
#define SRC_ECSIM_SIMULATION_H

#include <petscksp.h>

#include "src/interfaces/simulation.h"
#include "src/impls/ecsim/particles.h"
#include "src/utils/sync_clock.h"

Vector3R interpolate_E_s1(Arr E_g, const Vector3R& coord);
Vector3R interpolate_B_s1(Arr B_g, const Vector3R& coord);

namespace ecsim {

/// @note The following is a recreation of the published results,
/// @see https://doi.org/10.1016/j.jcp.2017.01.002
class Simulation : public interfaces::Simulation {
public:
  Simulation() = default;
  PetscErrorCode finalize() override;

  Vec Ep;
  Vec currI;

  /**
   * @details Filling up the Lapenta's matrix will be one of the most time
   * consuming part of the simulation process, so a proper way to do so should
   * be chosen. In attempts to speed up this process, we use the following steps:
   *
   * 1) `MatSetPreallocationCOO()`/`MatSetValuesCOO()` technique is utilized;
   * 2) Control of matrix indices assembly is added (to avoid reallocation);
   * 3) Parallel-traversable buffers are used to fill indices and values.
   *
   * @todo Check whether its faster to use `Vec` + `MATOP_MULT` for this.
   * @todo The only problem with storing matrix is vector is remapping between local and global
   */
  Mat matL;
  Mat matA;

  std::vector<std::shared_ptr<Particles>> particles_;

  void get_array_offset(PetscInt begin_g, PetscInt end_g, PetscInt& off);

  PetscErrorCode set_tolerances(
    PetscReal atol, PetscReal rtol, PetscReal divtol, PetscInt maxit);

protected:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(PetscInt t) override;

  virtual PetscErrorCode init_particles();
  virtual PetscErrorCode init_vectors();
  virtual PetscErrorCode init_ksp_solvers();
  virtual PetscErrorCode init_log_stages();
  PetscErrorCode init_matrices();

  // The main simulation steps
  PetscErrorCode clear_sources();
  PetscErrorCode first_push();
  PetscErrorCode advance_fields();
  PetscErrorCode second_push();
  PetscErrorCode final_update();

  PetscErrorCode advance_fields(KSP ksp, Vec curr, Vec out);
  PetscErrorCode update_cells_with_assembly();
  PetscErrorCode fill_ecsim_current();

  PetscErrorCode fill_matrix_indices(PetscInt* coo_i, PetscInt* coo_j);
  PetscErrorCode fill_ecsim_current(PetscReal* coo_v);

  PetscReal atol = 1e-7;
  PetscReal rtol = 1e-5;
  PetscReal divtol = PETSC_DETERMINE;
  PetscInt maxit = 1000;

  KSP ksp;

  /**
   * @brief Radius of the cloud, where indices should be assembled.
   * @details The available values are:
   * 1) radius = 0 -- All cells where _were_ particles are assembled.
   * 2) radius > 0 -- Adds a `radius` cells along each direction to
   *                  where the particles _were_ placed.
   * 3) radius < 0 -- Drop the previous `assembly_map` in this case.
   */
  static constexpr PetscInt assembly_radius = +1;
  static constexpr PetscInt assembly_width = 2 * assembly_radius + 1;

  /// @brief Cells, where indices have been assembled.
  /// @note This map and `matL` size can only _grow_ in time.
  std::vector<bool> assembly_map;

  /// @brief Whether the new cells have been added into `indices_map`.
  PetscInt indices_assembled = false;

  std::vector<PetscReal> conv_hist;

  PetscClassId classid;
  PetscLogEvent events[2];
  PetscLogStage stagenums[6];

  SyncClock clock;
};

}  // namespace ecsim

#endif  // SRC_ECSIM_SIMULATION_H
