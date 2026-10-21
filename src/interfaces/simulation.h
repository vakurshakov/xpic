#ifndef SRC_INTERFACES_SIMULATION_H
#define SRC_INTERFACES_SIMULATION_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/diagnostic.h"
#include "src/interfaces/particles.h"
#include "src/utils/world.h"

class Energy;

namespace interfaces {

class Simulation {
  friend class ::Energy;

public:
  Simulation() = default;
  virtual ~Simulation() = default;

  PetscInt start = 0;

  /**
   * @brief Contains the `DMDA` that should be used to create vectors and matrices
   * along with some addition information e.g. local sizes from `DMDAGetCorners()`.
   *
   * @note Before using it, `World::initialize()` should be used in some form.
   */
  World world;

  DM da;
  DM da_rho;

  Vec E;
  Vec E_loc;
  Arr E_arr;

  Vec B;
  Vec B_loc;
  Arr B_arr;

  Vec B0;

  Vec J;
  Vec J_loc;
  Arr J_arr;

  Mat rotE;
  Mat rotB;
  Mat matM;  // (I + (0.25 * dt * dt) rotB * rotE)
  Mat divE;

  // Utility vectors (e.g. to store rotE * E)
  Vec rE, rB;

  /**
   * @brief If false, the fields are prescribed by presets and stay constant,
   * only the particles are advanced (test-particle mode). Configured by
   * `"Fields": {"evolve": false}`.
   */
  bool evolve_fields = true;

  /**
   * @brief Container of abstract particles, the down-casted pointers to this
   * particles are stored in each `interfaces::Simulation` inheritor. The actual
   * memory for them is shared between `this` and inheritors.
   *
   * @note It is used for general-purpose operations with particles, e.g.
   * `Simulation::get_named_particles()` as it returns `interfaces::Particles&`.
   */
  std::vector<std::shared_ptr<Particles>> particles_;

  PetscErrorCode initialize();
  PetscErrorCode calculate();

  /// @brief Finishes diagnostics and releases simulation resources.
  virtual PetscErrorCode finalize();

  Particles& get_named_particles(std::string_view name);

protected:
  template<class SimSpec, class PartSpec>
  PetscErrorCode init_particles(SimSpec& that,  //
    std::vector<std::shared_ptr<PartSpec>>& container);

  PetscErrorCode log_information() const;

  /// @brief Creates fields, particle sorts and diagnostics of the concrete model.
  virtual PetscErrorCode initialize_implementation() = 0;

  /// @brief Advances fields and particles by one timestep according to the concrete model.
  virtual PetscErrorCode timestep_implementation(PetscInt timestep) = 0;

  std::vector<Command_up> step_presets_;
  std::vector<Diagnostic_up> diagnostics_;

  PetscLogStage stagenums[2];
};

}  // namespace interfaces

/// @note Contains the definitions for template methods in `interfaces::Simulation`.
#include "simulation.tpp"

/// @returns Concrete simulation using `config` specification.
std::unique_ptr<interfaces::Simulation> build_simulation();

#endif  // SRC_INTERFACES_SIMULATION_H
