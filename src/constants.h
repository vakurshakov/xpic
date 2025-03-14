#ifndef SRC_CONSTANTS_H
#define SRC_CONSTANTS_H

#define PARTICLES_FORM_FACTOR 2
#define RANDOM_SEED           false
#define LOGGING               true

#include <petscsystypes.h>

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

#endif  // SRC_CONSTANTS_H
