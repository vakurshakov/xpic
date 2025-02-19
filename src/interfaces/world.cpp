#include "src/interfaces/world.h"

#include "src/interfaces/sort_parameters.h"
#include "src/utils/configuration.h"

PetscErrorCode World::initialize()
{
  PetscFunctionBegin;
  const PetscInt dof = Vector3R::dim;
  const auto s = static_cast<PetscInt>(std::ceil(shape_radius));

  Configuration::get_boundaries_type(REP3_A(bounds));
  Configuration::get_processors(REP3_A(procs));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, REP3_A(bounds), DMDA_STENCIL_BOX, REP3_A(Geom_n), REP3_A(procs), dof, s, REP3(nullptr), &da));
  PetscCall(DMSetUp(da));

  PetscCall(DMDAGetNeighbors(da, &neighbors));

  PetscCall(DMDAGetCorners(da, REP3_A(&start), REP3_A(&size)));
  end = start + size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

World::~World()
{
  if (da)
    PetscCallVoid(DMDestroy(&da));
}
