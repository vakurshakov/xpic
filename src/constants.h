#ifndef SRC_CONSTANTS_H
#define SRC_CONSTANTS_H

#define PARTICLES_FORM_FACTOR 2

#define LOGGING true

#include <petscsystypes.h>
using timestep_t = PetscInt;

namespace physical_constants {

constexpr PetscReal e = 1.0;       // units of e
constexpr PetscReal me = 1.0;      // units of me
constexpr PetscReal Mp = 1836.0;   // units of me
constexpr PetscReal mec2 = 511.0;  // KeV

}

extern PetscReal Dx[3];           // c/w_pe
extern PetscReal dx;              // c/w_pe
extern PetscReal dy;              // c/w_pe
extern PetscReal dz;              // c/w_pe
extern PetscReal dt;              // 1/w_pe

extern PetscReal Geom[3];         // c/w_pe
extern PetscReal geom_x;          // c/w_pe
extern PetscReal geom_y;          // c/w_pe
extern PetscReal geom_z;          // c/w_pe
extern PetscReal geom_t;          // 1/w_pe

extern PetscInt Geom_n[3];        // units of [dx, dy, dz] accordingly
extern PetscInt geom_nx;          // units of dx
extern PetscInt geom_ny;          // units of dy
extern PetscInt geom_nz;          // units of dz
extern PetscInt geom_nt;          // units of dt

extern PetscInt diagnose_period;  // units of dt

void set_world_geometry( //
  PetscReal _gx, PetscReal _gy, PetscReal _gz, PetscReal _gt, //
  PetscReal _dx, PetscReal _dy, PetscReal _dz, PetscReal _dt, //
  PetscInt _dtp);

void set_world_geometry( //
  PetscInt _gnx, PetscInt _gny, PetscInt _gnz, PetscInt _gnt, //
  PetscReal _dx, PetscReal _dy, PetscReal _dz, PetscReal _dt, //
  PetscInt _dtp);

#endif  // SRC_CONSTANTS_H
