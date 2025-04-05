#ifndef SRC_INTERFACES_SIMULATION_H
#define SRC_INTERFACES_SIMULATION_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/diagnostic.h"
#include "src/interfaces/particles.h"
#include "src/utils/world.h"

class EnergyConservation;

namespace interfaces {

class Simulation {
  friend class ::EnergyConservation;

public:
  DEFAULT_MOVABLE(Simulation);

  Simulation() = default;
  virtual ~Simulation() = default;

  /// @brief From what timepoint the simulation starts, see `Simulation::calculate()`.
  PetscInt start = 0;

  /**
   * @brief Contains the `DMDA` that should be used to create vectors and matrices
   * along with some addition information e.g. local sizes from `DMDAGetCorners()`.
   *
   * @note Before using it, `World::initialize()` should be used in some form.
   */
  World world;

  /**
   * @brief Container of abstract particles, the down-casted pointers to this
   * particles are stored in each `interfaces::Simulation` inheritor. The actual
   * memory for them is shared between `this` and inheritor.
   *
   * @note It is used for general-purpose operations with particles, e.g.
   * `Simulation::get_named_particles()` as it returns `interfaces::Particles&`.
   */
  std::vector<std::shared_ptr<Particles>> particles_;

  PetscErrorCode initialize();
  PetscErrorCode calculate();

  virtual Vec get_named_vector(std::string_view name) const = 0;
  Particles& get_named_particles(std::string_view name) const;

  /// @todo Replace it with `std::vector<Vec>` and `PetscObjectSetName()`
  template<typename T>
  using NamedValues = std::map<std::string, T>;

  virtual NamedValues<Vec> get_backup_fields() const = 0;
  NamedValues<Particles*> get_backup_particles() const;

protected:
  template<class SimSpec, class PartSpec>
  PetscErrorCode init_particles(SimSpec& that,  //
    std::vector<std::shared_ptr<PartSpec>>& container);

  PetscErrorCode log_information() const;

  virtual PetscErrorCode initialize_implementation() = 0;
  virtual PetscErrorCode timestep_implementation(PetscInt timestep) = 0;

  std::vector<Command_up> step_presets_;
  std::vector<Diagnostic_up> diagnostics_;

  PetscLogStage stagenums[2];
};

}  // namespace interfaces

/// @note Contains the definitions for template methods in `interfaces::Simulation`.
#include "simulation.tpp"

using Simulation_up = std::unique_ptr<interfaces::Simulation>;

/// @returns Concrete simulation using `config` specification.
Simulation_up build_simulation();

#endif  // SRC_INTERFACES_SIMULATION_H
