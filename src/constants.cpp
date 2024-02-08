#include "constants.h"

PetscReal dx = 0.0;  // c / w_pe
PetscReal dy = 0.0;  // c / w_pe
PetscReal dz = 0.0;  // c / w_pe
PetscReal dt = 0.0;  // 1 / w_pe

PetscReal geom_x = 0.0;  // c / w_pe
PetscReal geom_y = 0.0;  // c / w_pe
PetscReal geom_z = 0.0;  // c / w_pe
PetscReal geom_t = 0.0;  // 1 / w_pe

PetscInt geom_nx = 0;  // units of dx
PetscInt geom_ny = 0;  // units of dy
PetscInt geom_nz = 0;  // units of dz
PetscInt geom_nt = 0;  // units of dt

PetscInt diagnose_period = 0;  // units of dt
