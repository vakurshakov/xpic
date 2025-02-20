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

  const PetscMPIInt* neighbors;

  Vector3R start;
  Vector3I start_n;

  Vector3R end;
  Vector3I end_n;
};

#endif  // SRC_BASIC_SIMULATION_H
