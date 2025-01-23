#ifndef SRC_INTERFACES_SIMULATION_H
#define SRC_INTERFACES_SIMULATION_H

#include "src/pch.h"
#include "src/interfaces/command.h"
#include "src/interfaces/diagnostic.h"
#include "src/interfaces/particles.h"
#include "src/interfaces/world.h"

namespace interfaces {

class Simulation {
public:
  DEFAULT_MOVABLE(Simulation);

  Simulation() = default;
  virtual ~Simulation() = default;

  World world_;

  PetscErrorCode initialize();
  PetscErrorCode calculate();

  virtual Vec get_named_vector(std::string_view name) = 0;
  virtual Particles& get_named_particles(std::string_view name) = 0;

protected:
  virtual PetscErrorCode initialize_implementation() = 0;
  virtual PetscErrorCode log_information() const;

  virtual PetscErrorCode timestep_implementation(timestep_t timestep) = 0;

  timestep_t start_ = 0;

  std::list<Command_up> step_presets_;
  std::vector<Diagnostic_up> diagnostics_;

  PetscLogStage stagenums[2];

  template<class ParticlesContainer>
  static auto&& get_named_particles(
    std::string_view name, ParticlesContainer& storage)
  {
    auto it = std::find_if(storage.begin(), storage.end(),  //
      [&](const auto& sort) {
        return sort.parameters().sort_name == name;
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
