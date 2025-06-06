#include "src/utils/world.h"

#include "src/interfaces/builder.h"
#include "src/utils/configuration.h"

World::World()
  : procs(REP3(PETSC_DECIDE)), bounds(REP3(DM_BOUNDARY_NONE))
{
}

PetscErrorCode World::initialize()
{
  PetscFunctionBeginUser;
  if (!CONFIG().json.empty()) {
    const Configuration::json_t& geometry = CONFIG().json.at("Geometry");

    // Reading cell sizes first to be able to use them in `Builder::parse_value()`
    Dx[0] = dx = geometry.at("dx").get<PetscReal>();
    Dx[1] = dy = geometry.at("dy").get<PetscReal>();
    Dx[2] = dz = geometry.at("dz").get<PetscReal>();
    dt = geometry.at("dt").get<PetscReal>();

    using namespace interfaces;

    set_geometry(  //
      Builder::parse_value(geometry.at("x")),
      Builder::parse_value(geometry.at("y")),
      Builder::parse_value(geometry.at("z")),
      Builder::parse_value(geometry.at("t")), dx, dy, dz, dt,
      Builder::parse_value(geometry.at("diagnose_period")));

    Configuration::get_processors(REP3_A(procs));
    Configuration::get_boundaries_type(REP3_A(bounds));
  }

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, REP3_A(bounds), DMDA_STENCIL_BOX, REP3_A(Geom_n), REP3_A(procs), dof, st, REP3(nullptr), &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMDAGetNeighbors(da, &neighbors));

  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));
  end = start + size;

  PetscCall(DMDAGetGhostCorners(da, REP3_A(&gstart), REP3_A(&gsize)));
  gend = gstart + gsize;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode World::finalize()
{
  PetscFunctionBeginUser;
  if (da)
    PetscCall(DMDestroy(&da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

World::~World()
{
  PetscCallAbort(PETSC_COMM_WORLD, finalize());
}


/* static */ void World::set_geometry(  //
  PetscReal _gx, PetscReal _gy, PetscReal _gz, PetscReal _gt,  //
  PetscReal _dx, PetscReal _dy, PetscReal _dz, PetscReal _dt,  //
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

/* static */ void World::set_geometry(  //
  PetscInt _gnx, PetscInt _gny, PetscInt _gnz, PetscInt _gnt,  //
  PetscReal _dx, PetscReal _dy, PetscReal _dz, PetscReal _dt,  //
  PetscReal _dtp)
{
  set_geometry(  //
    static_cast<PetscReal>(_gnx) * _dx,  //
    static_cast<PetscReal>(_gny) * _dy,  //
    static_cast<PetscReal>(_gnz) * _dz,  //
    static_cast<PetscReal>(_gnt) * _dt,  //
    _dx, _dy, _dz, _dt, _dtp);
}

/* static */ void World::set_geometry(  //
  PetscReal _gx, PetscReal _gy, PetscReal _gz, PetscReal _gt,  //
  PetscInt _gnx, PetscInt _gny, PetscInt _gnz, PetscInt _gnt,  //
  PetscReal _dtp)
{
  set_geometry(  //
    _gx, _gy, _gz, _gt,  //
    _gx / static_cast<PetscReal>(_gnx),  //
    _gy / static_cast<PetscReal>(_gny),  //
    _gz / static_cast<PetscReal>(_gnz),  //
    _gt / static_cast<PetscReal>(_gnt),  //
    _dtp);
}
