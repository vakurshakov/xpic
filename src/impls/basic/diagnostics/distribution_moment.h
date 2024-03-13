#ifndef SRC_BASIC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
#define SRC_BASIC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H

#include "src/interfaces/diagnostic.h"

#include <petscdmda.h>
#include <petscvec.h>

#include "src/pch.h"
#include "src/impls/basic/particles.h"
#include "src/utils/mpi_binary_file.h"

namespace basic {

struct Moment;
struct Projector;

using Moment_up = std::unique_ptr<Moment>;
using Projector_up = std::unique_ptr<Projector>;

class Distribution_moment : public interfaces::Diagnostic {
public:
  struct Region {
    Vector3<PetscInt> min;
    Vector3<PetscInt> max;
    Vector3<PetscReal> dp;
  };

  Distribution_moment(
    const std::string& result_directory, const DM& da, const Particles& particles,
    Moment_up moment, Projector_up projector);

  PetscErrorCode diagnose(timestep_t t) override;

protected:
  PetscErrorCode collect();
  PetscErrorCode clear();

  const DM& da_;
  const Particles& particles_;

  Vec data_;
  Region region_;

  Moment_up moment_;
  Projector_up projector_;

  MPI_Comm comm_;
  MPI_binary_file file_;
};


/**
 * @brief An utility structure to store the moment name
 * along with the pointer to distribution moment getter.
 *
 * Getter is chosen depending on the name in the constructor.
 */
struct Moment {
  Moment(const Particles& particles, const std::string& name);

  const Particles& particles_;

  using getter = PetscReal(*)(const Particles&, const Point&);
  getter get = nullptr;
};


/**
 * @brief Stores geometry: min and max value of a
 * projection, its step dp and two projectors to
 * axes.
 *
 * Projectors are selected depending on the name in the constructor.
 */
struct Projector {
  Projector(const Particles& particles, const std::string& axes_names);

  const Particles& particles_;

  using getter = PetscReal(*)(const Particles&, const Point&);
  getter get_x = nullptr;
  getter get_y = nullptr;
};

}

#endif  // SRC_BASIC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
