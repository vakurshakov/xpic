#include "src/utils/world.h"

#include "src/interfaces/builder.h"
#include "src/utils/configuration.h"
#include "src/utils/operators.h"
#include "src/utils/vector_utils.h"

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

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, REP3_A(bounds), DMDA_STENCIL_BOX, REP3_A(Geom_n), REP3_A(procs), dof, s, REP3(nullptr), &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  Region region{
    .dim = 3,
    .dof = 1,
    .start = Vector4I(0, 0, 0, 0),
    .size = Vector4I(geom_nx, geom_ny, geom_nz, 1),
  };
  PetscCall(create_local_dm(da, region, PETSC_COMM_WORLD, &da_rho));

  PetscCall(DMSetApplicationContext(da, this));
  PetscCall(DMSetApplicationContext(da_rho, this));

  PetscCall(DMDAGetInfo(da, NULL, REP3(NULL), REP3_A(&procs), NULL, NULL, REP3(NULL), NULL));
  PetscCall(DMDAGetNeighbors(da, &neighbors));
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));
  PetscCall(DMDAGetGhostCorners(da, REP3_A(&gstart), REP3_A(&gsize)));
  PetscCall(DMDAGetOwnershipRanges(da, REP3_A(&lg)));

  end = start + size;
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


/**
 * @returns Non-null communicator for those processes,
 * where region intersects with local boundaries of DM.
 */
PetscErrorCode World::create_local_comm(
  DM da, const Region& region, MPI_Comm* local_comm)
{
  PetscFunctionBeginUser;
  Vector3I r_start(region.start);
  Vector3I r_size(region.size);
  Vector3I start;
  Vector3I size;
  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));

  PetscMPIInt color =
    is_region_intersect_bounds(r_start, r_size, start, size) ? 1 : MPI_UNDEFINED;
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, color, rank, local_comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode World::create_local_dm(
  DM da, const Region& region, MPI_Comm local_comm, DM* local_da)
{
  PetscFunctionBeginUser;
  Vector3I g_start = vector_cast(region.start);
  Vector3I g_size = vector_cast(region.size);
  Vector3I g_end = g_start + g_size;

  PetscInt s;
  DMDAStencilType st;
  PetscInt size[3];
  PetscInt proc[3];
  DMBoundaryType bound[3];
  PetscCall(DMDAGetInfo(da, nullptr, REP3_A(&size), REP3_A(&proc), nullptr, &s, REP3_A(&bound), &st));

  const PetscInt* ownership[3];
  PetscCall(DMDAGetOwnershipRanges(da, REP3_A(&ownership)));

  PetscInt l_proc[3];
  DMBoundaryType l_bound[3];
  std::vector<PetscInt> l_ownership[3];

  // Collecting number of processes and ownership ranges using global DMDA
  for (PetscInt i = 0; i < 3; ++i) {
    l_proc[i] = 0;

    PetscInt start = 0, end = 0;

    for (PetscInt s = 0; s < proc[i]; ++s) {
      end += ownership[i][s];

      if (g_start[i] < end && start < g_end[i]) {
        l_proc[i]++;

        PetscInt l_si = std::max(g_start[i], start);
        PetscInt l_ei = std::min(g_end[i], end);

        l_ownership[i].emplace_back(l_ei - l_si);
      }

      start += ownership[i][s];
    }

    // Mimic global boundaries, if we touch them
    l_bound[i] = (g_size[i] == size[i]) ? bound[i] : DM_BOUNDARY_GHOSTED;
  }

  PetscCall(DMDACreate3d(local_comm, REP3_A(l_bound), st, REP3_A(g_size), REP3_A(l_proc), region.dof, s, l_ownership[X].data(), l_ownership[Y].data(), l_ownership[Z].data(), local_da));
  PetscCall(DMDASetOffset(*local_da, REP3_A(g_start), REP3(0)));
  PetscCall(DMSetUp(*local_da));
  PetscFunctionReturn(PETSC_SUCCESS);
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
