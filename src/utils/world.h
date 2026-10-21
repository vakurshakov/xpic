#ifndef SRC_UTILS_WORLD_H
#define SRC_UTILS_WORLD_H

#include <petscdmda.h>

#include "src/pch.h"
#include "src/interfaces/sort_parameters.h"
#include "src/utils/vector3.h"
#include "src/utils/vector4.h"

struct Region {
  PetscInt dim;
  PetscInt dof;
  Vector4I start;
  Vector4I size;
};

struct World {
  World();
  ~World();

  PetscErrorCode initialize();
  PetscErrorCode finalize();

  DM da = NULL;
  DM da_rho = NULL;

  PetscInt procs[3];
  DMBoundaryType bounds[3];

  PetscInt neighbors_num;
  const PetscMPIInt* neighbors;

  const PetscInt dof = Vector3R::dim;
  const PetscInt s = 2 * static_cast<PetscInt>(std::ceil(shape_radius));
  const DMDAStencilType st = DMDA_STENCIL_BOX;
  const PetscInt* lg[3];

  Vector3I start;
  Vector3I size;
  Vector3I end;

  Vector3I gstart;
  Vector3I gsize;
  Vector3I gend;

  static PetscErrorCode create_local_comm(
    DM da, const Region& region, MPI_Comm* local_comm);

  static PetscErrorCode create_local_dm(
    DM da, const Region& region, MPI_Comm local_comm, DM* local_da);

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
