#ifndef SRC_COMMANDS_INJECT_PARTICLES_H
#define SRC_COMMANDS_INJECT_PARTICLES_H

#include "src/commands/set_particles.h"

/// @brief Loads the ionized and the ejected sort (in that order)
/// every step within `[injection_start, injection_end]`.
class InjectParticles : public SetParticles {
public:
  InjectParticles(                                    //
    std::vector<Sort> sorts,                          //
    PetscInt injection_start,                         //
    PetscInt injection_end,                           //
    PetscInt per_step_particles_num,                  //
    const CoordinateGenerator& generate_coordinate);  //

  PetscErrorCode execute(PetscInt t) override;

  std::string get_ionized_name() const;
  std::string get_ejected_name() const;
  PetscReal get_ionized_energy() const;
  PetscReal get_ejected_energy() const;

private:
  PetscInt injection_start;
  PetscInt injection_end;
};

#endif  // SRC_COMMANDS_INJECT_PARTICLES_H
