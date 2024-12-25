#ifndef SRC_INTERFACES_PARTICLES_H
#define SRC_INTERFACES_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/point.h"
#include "src/interfaces/sort_parameters.h"
#include "src/interfaces/world.h"

namespace interfaces {

class Particles {
public:
  DEFAULT_MOVABLE(Particles);

  Particles(const World& world, const SortParameters& parameters);
  virtual ~Particles() = default;

  const World& world_;

  void reserve(PetscInt number_of_particles);
  PetscErrorCode add_particle(const Point& point);

  const SortParameters& parameters() const;

  std::vector<Point>& points();
  const std::vector<Point>& points() const;

  PetscReal mass(const Point& point) const;
  PetscReal charge(const Point& point) const;
  PetscReal density(const Point& point) const;
  PetscInt particles_number(const Point& point) const;

  Vector3R velocity(const Point& point) const;

  PetscErrorCode communicate();

protected:
  static constexpr PetscInt MPI_TAG_NUMBERS = 2;
  static constexpr PetscInt MPI_TAG_POINTS = 4;

  SortParameters parameters_;
  std::vector<Point> points_;
};

}  // namespace interfaces

using Particles_up = std::unique_ptr<interfaces::Particles>;

#endif  // SRC_INTERFACES_PARTICLES_H
