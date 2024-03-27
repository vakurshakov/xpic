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
using Moment_up = std::unique_ptr<Moment>;

class Distribution_moment : public interfaces::Diagnostic {
public:
  struct Region {
    Vector3<PetscInt> start;
    Vector3<PetscInt> size;
    Vector3<PetscReal> dp;
  };

  Distribution_moment(MPI_Comm comm, const std::string& result_directory,
    const DM& da, const Particles& particles, Moment_up moment);

  PetscErrorCode set_diagnosed_region(const Region& region);
  PetscErrorCode diagnose(timestep_t t) override;

private:
  PetscErrorCode collect();
  PetscErrorCode clear();

  PetscInt index(PetscInt x, PetscInt y, PetscInt z) const {
    return
      ((z - region_.start[Z]) * region_.size[Y] +
      (y - region_.start[Y])) * region_.size[X] +
      (x - region_.start[X]);
  }

private:
  const DM& da_;
  const Particles& particles_;

  std::vector<PetscReal> data_;
  Region region_;

  Moment_up moment_;

  MPI_Comm comm_;
  MPI_binary_file file_;
};


/**
 * @brief An utility structure to store the pointer to distribution moment getter.
 * Getter is chosen depending on the name in the constructor.
 */
struct Moment {
  Moment(const Particles& particles, const std::string& name);

  const Particles& particles_;

  using getter = PetscReal(*)(const Particles&, const Point&);
  getter get = nullptr;
};

}

#endif  // SRC_BASIC_DIAGNOSTICS_DISTRIBUTION_MOMENT_H
