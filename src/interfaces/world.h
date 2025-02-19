#ifndef SRC_INTERFACES_WORLD_H
#define SRC_INTERFACES_WORLD_H

#include <petscdmda.h>

#include "src/pch.h"
#include "src/utils/vector3.h"

struct World {
  DEFAULT_MOVABLE(World);

  World() = default;
  ~World();

  PetscErrorCode initialize();

  DM da = nullptr;
  PetscInt procs[3];
  DMBoundaryType bounds[3];

  PetscInt neighbors_num;
  const PetscMPIInt* neighbors;

  Vector3I start;
  Vector3I size;
  Vector3I end;

  inline PetscInt s_g(PetscInt z, PetscInt y, PetscInt x) const
  {
    return indexing::petsc_index(z, y, x, 0, size[Z], size[Y], size[X], 1);
  }

  inline PetscInt v_g(PetscInt z, PetscInt y, PetscInt x, PetscInt c) const
  {
    return indexing::petsc_index(z, y, x, c, size[Z], size[Y], size[X], 3);
  }
};

#endif  // SRC_BASIC_SIMULATION_H
