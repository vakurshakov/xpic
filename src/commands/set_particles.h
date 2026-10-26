#ifndef SRC_COMMANDS_SET_PARTICLES_H
#define SRC_COMMANDS_SET_PARTICLES_H

#include "src/interfaces/command.h"
#include "src/interfaces/particles.h"
#include "src/utils/particles_load.h"

class SetParticles : public interfaces::Command {
public:
  struct Sort {
    interfaces::Particles& particles;
    MomentumGenerator generate_momentum;
  };

  /// @brief All sorts share the generated coordinate, each has its own momentum.
  SetParticles(                                       //
    std::vector<Sort> sorts,                          //
    PetscInt number_of_particles,                     //
    const CoordinateGenerator& generate_coordinate);  //

  PetscErrorCode execute(PetscInt t) override;

protected:
  PetscErrorCode load(PetscInt number_of_particles);
  PetscErrorCode log_statistics();

  std::vector<Sort> sorts;
  CoordinateGenerator generate_coordinate;

  std::vector<PetscReal> added_energy;
  std::vector<PetscInt> added_particles;

  /// @todo Number of loaded particles can vary, if `MPI_Comm_size() > 1`
  PetscInt number_of_particles;
};

#endif  // SRC_COMMANDS_SET_PARTICLES_H
