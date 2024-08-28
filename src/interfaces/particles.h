#ifndef SRC_INTERFACES_PARTICLES_H
#define SRC_INTERFACES_PARTICLES_H

#include "src/pch.h"

#include <petscdm.h>

#include "src/utils/vector3.h"
#include "src/interfaces/point.h"
#include "src/interfaces/sort_parameters.h"

namespace interfaces {

class Particles {
public:
  virtual ~Particles() = default;

  const Sort_parameters& parameters() const;

  PetscInt particles_number(const Point& point) const;
  PetscReal density(const Point& point) const;
  PetscReal charge(const Point& point) const;
  PetscReal mass(const Point& point) const;

  Vector3R velocity(const Point& point) const;

protected:
  static constexpr int MPI_TAG_NUMBERS = 2;
  static constexpr int MPI_TAG_POINTS  = 4;

  Sort_parameters parameters_;
};

}

#endif  // SRC_INTERFACES_PARTICLES_H
