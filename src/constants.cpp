#include "constants.h"

double dx = 0.0;  // c / w_pe
double dy = 0.0;  // c / w_pe
double dz = 0.0;  // c / w_pe
double dt = 0.0;  // 1 / w_pe

int geom_nx = 0;         // units of dx
int geom_ny = 0;         // units of dy
int geom_nz = 0;         // units of dz
timestep_t geom_nt = 0;  // units of dt

double geom_x = 0.0;  // c / w_pe
double geom_y = 0.0;  // c / w_pe
double geom_z = 0.0;  // c / w_pe
double geom_t = 0.0;  // 1 / w_pe

timestep_t diagnose_period = 0;  // units of dt
