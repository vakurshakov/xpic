#include "src/interfaces/world.h"

#include "src/utils/particle_shape.h"

void convert(Vector3R& vector, const Vector3I& other) {
  vector.x() = (PetscReal)other.x() * dx;
  vector.y() = (PetscReal)other.y() * dy;
  vector.z() = (PetscReal)other.z() * dz;
}

PetscErrorCode World::initialize() {
  PetscFunctionBegin;
  const PetscInt dof = Vector3R::dim;
  const PetscInt s = shape_radius;

  Configuration::get_boundaries_type(REP3_A(bounds));
  Configuration::get_processors(REP3_A(procs));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, REP3_A(bounds), DMDA_STENCIL_BOX, REP3_A(Geom_n), REP3_A(procs), dof, s, REP3(nullptr), &da));
  PetscCall(DMSetUp(da));

  PetscCall(DMDAGetNeighbors(da, &neighbors));

  PetscCall(DMDAGetCorners(da, REP3_A(&start_n), REP3_A(&size_n)));
  convert(start, start_n);

  size_n.x() = std::min(shape_width, geom_nx);
  size_n.y() = std::min(shape_width, geom_ny);
  size_n.z() = std::min(shape_width, geom_nz);
  convert(size, size_n);

  end_n.x() = start_n.x() + size_n.x();
  end_n.y() = start_n.y() + size_n.y();
  end_n.z() = start_n.z() + size_n.z();
  convert(end, end_n);
  PetscFunctionReturn(PETSC_SUCCESS);
}

World::~World() {
  PetscCallVoid(DMDestroy(&da));
}