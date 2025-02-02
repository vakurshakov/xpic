#ifndef SRC_INTERFACES_PARTICLES_H
#define SRC_INTERFACES_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/point.h"
#include "src/interfaces/sort_parameters.h"
#include "src/interfaces/world.h"

namespace interfaces {

/// @note Points are stored in a contiguous 3d array of cells
class Particles {
public:
  DEFAULT_MOVABLE(Particles);

  Particles(const World& world, const SortParameters& parameters);
  virtual ~Particles() = default;

  const World& world;
  const SortParameters parameters;
  std::vector<std::list<Point>> storage;

  PetscErrorCode add_particle(const Point& point);
  PetscErrorCode correct_coordinates();
  // PetscErrorCode communicate();

  virtual PetscErrorCode update_cells();

  PetscReal mass(const Point& point) const;
  PetscReal charge(const Point& point) const;
  PetscReal density(const Point& point) const;
  PetscInt particles_number(const Point& point) const;

  Vector3R velocity(const Point& point) const;

protected:
  static constexpr PetscInt MPI_TAG_NUMBERS = 2;
  static constexpr PetscInt MPI_TAG_POINTS = 4;

  void correct_coordinates(Point& point);
};

}  // namespace interfaces

using Particles_up = std::unique_ptr<interfaces::Particles>;

#endif  // SRC_INTERFACES_PARTICLES_H
