#ifndef SRC_INTERFACES_PARTICLES_H
#define SRC_INTERFACES_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/point.h"
#include "src/interfaces/sort_parameters.h"
#include "src/utils/world.h"

namespace interfaces {

class Particles {
public:
  Particles(const World& world, const SortParameters& parameters);
  virtual ~Particles() = default;
  static int counter;

  Arr E_arr;
  Arr B_arr;

  Vec J;
  Vec J_loc;
  Arr J_arr;

  /// @brief Reference to outer `interfaces::Simulation::world`.
  const World& world;
  const DM da;

  /// @brief Particles parameters, constant throughout the whole simulation run.
  const SortParameters parameters;

  /// @brief Points are stored in a contiguous 3d array of cells.
  std::vector<std::list<Point>> storage;

  virtual PetscErrorCode finalize();

  /**
   * @brief The added `point` coordinate is first tested against local
   * MPI-boundaries. If it lies within it, the `point` is added to `storage`
   * and added `energy` is accumulated.
   *
   * @param point The inserted point to be emplaced into `storage`.
   * @param is_added Whether the `point` has been added after the MPI-boundaries test.
   */
  PetscErrorCode add_particle(const Point& point, bool* is_added = nullptr);

  PetscErrorCode correct_coordinates();
  PetscErrorCode correct_coordinates(Point& point);

  /**
   * @brief The following are the particle in cell carrying utils, the
   * proper one for the context would be set based on `MPI_Comm_size()`
   * within `Particles` constructor.
   */
  std::function<PetscErrorCode()> update_cells;
  PetscErrorCode update_cells_seq();
  PetscErrorCode update_cells_mpi();

  PetscReal mass(const Point& point) const;
  PetscReal charge(const Point& point) const;
  PetscReal density(const Point& point) const;
  PetscInt particles_number(const Point& point) const;

  PetscReal q_m(const Point& point) const;
  PetscReal n_Np(const Point& point) const;
  PetscReal qn_Np(const Point& point) const;

  Vector3R velocity(const Point& point) const;

  PetscErrorCode log_distribution() const;

protected:
  static constexpr PetscInt MPI_TAG_NUMBERS = 2;
  static constexpr PetscInt MPI_TAG_POINTS = 4;
  static constexpr PetscInt OMP_CHUNK_SIZE = 16;

  PetscClassId classid;
  PetscLogEvent events[1];
};

}  // namespace interfaces

using Particles_sp = std::shared_ptr<interfaces::Particles>;

#endif  // SRC_INTERFACES_PARTICLES_H
