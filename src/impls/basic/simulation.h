#ifndef SRC_BASIC_SIMULATION_H
#define SRC_BASIC_SIMULATION_H

#include "src/interfaces/simulation.h"
#include "src/impls/basic/particles.h"

namespace basic {

class Simulation : public interfaces::Simulation {
public:
  DEFAULT_MOVABLE(Simulation);

  Simulation() = default;
  ~Simulation() override;

  Vec E;
  Vec B;
  Vec J;
  std::vector<std::unique_ptr<Particles>> particles_;

  Vec get_named_vector(std::string_view name) override;
  Particles& get_named_particles(std::string_view name) override;

private:
  PetscErrorCode init_particles();
  PetscErrorCode push_particles();
  PetscErrorCode push_fields();

  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(PetscInt timestep) override;

  Vec local_E;
  Vec local_B;

  Mat rot_dt_p;
  Mat rot_dt_m;
};

}  // namespace basic

#endif  // SRC_BASIC_SIMULATION_H
