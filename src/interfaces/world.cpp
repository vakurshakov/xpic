#include "src/interfaces/world.h"

#include "src/interfaces/sort_parameters.h"
#include "src/utils/configuration.h"
#include "src/utils/utils.h"

void convert(Vector3R& vector, const Vector3I& other)
{
  vector.x() = static_cast<PetscReal>(other.x()) * dx;
  vector.y() = static_cast<PetscReal>(other.y()) * dy;
  vector.z() = static_cast<PetscReal>(other.z()) * dz;
}

PetscErrorCode World::initialize()
{
  PetscFunctionBegin;
  const PetscInt dof = Vector3R::dim;
  const PetscInt s = static_cast<PetscInt>(std::ceil(shape_radius));

  Configuration::get_boundaries_type(REP3_A(bounds));
  Configuration::get_processors(REP3_A(procs));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, REP3_A(bounds), DMDA_STENCIL_BOX, REP3_A(Geom_n), REP3_A(procs), dof, s, REP3(nullptr), &da));
  PetscCall(DMSetUp(da));

  PetscCall(DMDAGetNeighbors(da, &neighbors));

  PetscCall(DMDAGetCorners(da, REP3_A(&start_n), REP3_A(&end_n)));
  end_n.x() += start_n.x();
  end_n.y() += start_n.y();
  end_n.z() += start_n.z();

  convert(start, start_n);
  convert(end, end_n);
  PetscFunctionReturn(PETSC_SUCCESS);
}

World::~World()
{
  if (da)
    PetscCallVoid(DMDestroy(&da));
}
