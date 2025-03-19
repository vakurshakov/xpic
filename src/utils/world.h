#ifndef SRC_UTILS_WORLD_H
#define SRC_UTILS_WORLD_H

#include <petscdmda.h>

#include "src/pch.h"
#include "src/utils/vector3.h"

struct World {
  DEFAULT_MOVABLE(World);

  World();
  ~World();

  PetscErrorCode initialize();
  PetscErrorCode finalize();

  DM da = nullptr;
  PetscInt procs[3];
  DMBoundaryType bounds[3];

  PetscInt neighbors_num;
  const PetscMPIInt* neighbors;

  Vector3I start;
  Vector3I size;
  Vector3I end;

  inline PetscInt s_g(PetscInt x, PetscInt y, PetscInt z) const
  {
    return indexing::petsc_index(x, y, z, 0, size[X], size[Y], size[Z], 1);
  }

  inline PetscInt v_g(PetscInt x, PetscInt y, PetscInt z, PetscInt c) const
  {
    return indexing::petsc_index(x, y, z, c, size[X], size[Y], size[Z], 3);
  }

  static void set_geometry( //
    PetscReal _gx, PetscReal _gy, PetscReal _gz, PetscReal _gt, //
    PetscReal _dx, PetscReal _dy, PetscReal _dz, PetscReal _dt, //
    PetscReal _dtp);

  static void set_geometry( //
    PetscInt _gnx, PetscInt _gny, PetscInt _gnz, PetscInt _gnt, //
    PetscReal _dx, PetscReal _dy, PetscReal _dz, PetscReal _dt, //
    PetscReal _dtp);

  static void set_geometry( //
    PetscReal _gx, PetscReal _gy, PetscReal _gz, PetscReal _gt, //
    PetscInt _gnx, PetscInt _gny, PetscInt _gnz, PetscInt _gnt, //
    PetscReal _dtp);
};

#endif  // SRC_UTILS_WORLD_H
