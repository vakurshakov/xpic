#ifndef SRC_INTERFACES_WORLD_H
#define SRC_INTERFACES_WORLD_H

#include "src/pch.h"

#include <petscdmda.h>

struct World {
  World() = default;
  ~World();

  PetscErrorCode initialize();

  DM da;
  PetscInt procs[3];
  DMBoundaryType bounds[3];

  const PetscMPIInt* neighbours;

  Vector3R start;
  Vector3R size;
  Vector3R end;

  Vector3I start_n;
  Vector3I size_n;
  Vector3I end_n;
};

#endif // SRC_BASIC_SIMULATION_H