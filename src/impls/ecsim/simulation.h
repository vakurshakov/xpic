#ifndef SRC_ECSIM_SIMULATION_H
#define SRC_ECSIM_SIMULATION_H

#include <petscksp.h>

#include "src/interfaces/simulation.h"
#include "src/impls/ecsim/particles.h"
#include "src/utils/sync_clock.h"

namespace ecsim {

class Simulation : public interfaces::Simulation {
public:
  DEFAULT_MOVABLE(Simulation);

  Simulation() = default;
  ~Simulation() override;

  Vec E;
  Vec Eh;
  Vec B;
  Vec B0;
  Vec currI;

  Mat matL;

  std::vector<std::shared_ptr<Particles>> particles_;

  Vec get_named_vector(std::string_view name) const override;
  NamedValues<Vec> get_backup_fields() const override;

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(PetscInt t) override;

  PetscErrorCode init_vectors();
  PetscErrorCode init_matrices();
  PetscErrorCode init_ksp_solvers();

  // The main simulation steps
  PetscErrorCode clear_sources();
  PetscErrorCode first_push();
  PetscErrorCode fill_ecsim_current();
  PetscErrorCode predict_fields();
  PetscErrorCode second_push();
  PetscErrorCode final_update();

  PetscErrorCode update_cells_with_assembly();
  PetscErrorCode advance_fields(KSP ksp, Vec curr, Vec out);

  void get_array_offset(PetscInt begin_g, PetscInt end_g, PetscInt& off);

  PetscErrorCode fill_matrix_indices(PetscInt* coo_i, PetscInt* coo_j);
  PetscErrorCode fill_ecsim_current(PetscReal* coo_v);

  Vec local_E;
  Vec local_B;

  Mat rotE;
  Mat rotB;
  Mat matM;

  KSP ksp;

  static constexpr PetscInt assembly_radius = +1;
  static constexpr PetscInt assembly_width = 2 * assembly_radius + 1;

  std::vector<bool> assembly_map;

  bool indices_assembled = false;

  /// @todo Rework clock, there is a problem if you use e.g. __FUNCTION__
  /// SyncClock clock;

  friend class EnergyConservation;
};

}  // namespace ecsim

#endif  // SRC_ECSIM_SIMULATION_H
