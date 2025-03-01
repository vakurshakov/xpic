#include "constants.h"

#include "src/utils/utils.h"

PetscReal Dx[3] = {0.0, 0.0, 0.0};    // c/w_pe
PetscReal dx = 0.0;                   // c/w_pe
PetscReal dy = 0.0;                   // c/w_pe
PetscReal dz = 0.0;                   // c/w_pe
PetscReal dt = 0.0;                   // 1/w_pe

PetscReal Geom[3] = {0.0, 0.0, 0.0};  // c/w_pe
PetscReal geom_x = 0.0;               // c/w_pe
PetscReal geom_y = 0.0;               // c/w_pe
PetscReal geom_z = 0.0;               // c/w_pe
PetscReal geom_t = 0.0;               // 1/w_pe

PetscInt Geom_n[3] = {0, 0, 0};       // units of [dx, dy, dz] accordingly
PetscInt geom_nx = 0;                 // units of dx
PetscInt geom_ny = 0;                 // units of dy
PetscInt geom_nz = 0;                 // units of dz
PetscInt geom_nt = 0;                 // units of dt

PetscInt diagnose_period = 0;         // units of dt

void set_world_geometry( //
  PetscReal _gx, PetscReal _gy, PetscReal _gz, PetscReal _gt, //
  PetscReal _dx, PetscReal _dy, PetscReal _dz, PetscReal _dt, //
  PetscReal _dtp)
{
  Dx[0] = dx = _dx;
  Dx[1] = dy = _dy;
  Dx[2] = dz = _dz;
  dt = _dt;

  Geom[0] = geom_x = _gx;
  Geom[1] = geom_y = _gy;
  Geom[2] = geom_z = _gz;
  geom_t = _gt;

  Geom_n[0] = geom_nx = ROUND_STEP(geom_x, dx);
  Geom_n[1] = geom_ny = ROUND_STEP(geom_y, dy);
  Geom_n[2] = geom_nz = ROUND_STEP(geom_z, dz);
  geom_nt = ROUND_STEP(geom_t, dt);

  diagnose_period = ROUND_STEP(_dtp, dt);
}

void set_world_geometry( //
  PetscInt _gnx, PetscInt _gny, PetscInt _gnz, PetscInt _gnt, //
  PetscReal _dx, PetscReal _dy, PetscReal _dz, PetscReal _dt, //
  PetscReal _dtp)
{
  set_world_geometry( //
    static_cast<PetscReal>(_gnx) * _dx, //
    static_cast<PetscReal>(_gny) * _dy, //
    static_cast<PetscReal>(_gnz) * _dz, //
    static_cast<PetscReal>(_gnt) * _dt, //
    _dx, _dy, _dz, _dt, _dtp);
}
