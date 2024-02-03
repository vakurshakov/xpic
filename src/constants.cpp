#include "constants.h"

double dx = 0.0;  // c / w_pe
double dy = 0.0;  // c / w_pe
double dz = 0.0;  // c / w_pe
double dt = 0.0;  // 1 / w_pe

int size_nx = 0;         // units of dx
int size_ny = 0;         // units of dy
int size_nz = 0;         // units of dz
timestep_t size_nt = 0;  // units of dt

double size_lx = 0.0;  // c / w_pe
double size_ly = 0.0;  // c / w_pe
double size_lz = 0.0;  // c / w_pe
double size_lt = 0.0;  // 1 / w_pe

timestep_t diagnose_period = 0;  // units of dt
