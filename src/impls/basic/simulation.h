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

  Vec E_;
  Vec B_;
  Vec J_;
  std::vector<Particles> particles_;

  Vec get_named_vector(std::string_view name) const override;
  const Particles& get_named_particles(std::string_view name) const override;

private:
  PetscErrorCode initialize_implementation() override;
  PetscErrorCode timestep_implementation(timestep_t timestep) override;

  Mat rot_dt_p;
  Mat rot_dt_m;
};

}  // namespace basic

#endif  // SRC_BASIC_SIMULATION_H
