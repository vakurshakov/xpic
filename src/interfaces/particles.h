#ifndef SRC_INTERFACES_PARTICLES_H
#define SRC_INTERFACES_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/point.h"
#include "src/interfaces/sort_parameters.h"
#include "src/interfaces/world.h"

namespace interfaces {

class Particles {
public:
  Particles(const World& world, const Sort_parameters& parameters);
  virtual ~Particles() = default;

  const World& world_;

  PetscErrorCode add_particle(const Point& point);

  const Sort_parameters& parameters() const;
  const std::vector<Point>& points() const;

  PetscInt particles_number(const Point& point) const;
  PetscReal density(const Point& point) const;
  PetscReal charge(const Point& point) const;
  PetscReal mass(const Point& point) const;
  Vector3R velocity(const Point& point) const;

  PetscErrorCode communicate();

protected:
  static constexpr int MPI_TAG_NUMBERS = 2;
  static constexpr int MPI_TAG_POINTS = 4;

  PetscInt to_contiguous_index(PetscInt z, PetscInt y, PetscInt x)
  {
    constexpr PetscInt dim = 3;
    return (z * dim + y) * dim + x;
  }

  void from_contiguous_index(PetscInt index, PetscInt& z, PetscInt& y, PetscInt& x)
  {
    constexpr PetscInt dim = 3;
    x = (index) % dim;
    y = (index / dim) % dim;
    z = (index / dim) / dim;
  }

  Sort_parameters parameters_;
  std::vector<Point> points_;
};

}  // namespace interfaces

#endif  // SRC_INTERFACES_PARTICLES_H
