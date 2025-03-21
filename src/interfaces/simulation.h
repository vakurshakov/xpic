#ifndef SRC_INTERFACES_SIMULATION_H
#define SRC_INTERFACES_SIMULATION_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/diagnostic.h"
#include "src/interfaces/particles.h"
#include "src/utils/world.h"

namespace interfaces {

/// @todo Try to use a CRTP for Simulation and stub the default
/// implementation of particles to `interfaces::Particles`
 class Simulation {
public:
  DEFAULT_MOVABLE(Simulation);

  Simulation() = default;
  virtual ~Simulation() = default;

  World world;
  PetscInt start = 0;

  PetscErrorCode initialize();
  PetscErrorCode calculate();

  virtual Vec get_named_vector(std::string_view name) = 0;
  virtual Particles& get_named_particles(std::string_view name) = 0;

  template<typename T>
  using NamedValues = std::map<std::string, T>;

  virtual NamedValues<Vec> get_backup_fields()
  {
    return NamedValues<Vec>{};
  }

  virtual NamedValues<Particles*> get_backup_particles()
  {
    return NamedValues<Particles*>{};
  }

protected:
  virtual PetscErrorCode initialize_implementation() = 0;
  virtual PetscErrorCode timestep_implementation(PetscInt timestep) = 0;
  PetscErrorCode log_information() const;

  std::vector<Command_up> step_presets_;
  std::vector<Diagnostic_up> diagnostics_;

  PetscLogStage stagenums[2];

  template<class ParticlesContainer>
  static auto&& get_named_particles(
    std::string_view name, ParticlesContainer& storage)
  {
    auto it = std::find_if(storage.begin(), storage.end(),  //
      [&](const auto& sort) {
        return sort.parameters.sort_name == name;
      });

    if (it == storage.end())
      throw std::runtime_error("No particles with name " + std::string(name));
    return *it;
  }
};

}  // namespace interfaces

using Simulation_up = std::unique_ptr<interfaces::Simulation>;

/// @returns Concrete simulation using `config` specification
Simulation_up build_simulation();

#endif  // SRC_INTERFACES_SIMULATION_H
